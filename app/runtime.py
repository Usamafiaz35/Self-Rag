"""
Shared runtime bootstrap for both CLI and FastAPI.

Why this file exists:
- Keep initialization (settings, retriever, llm, graph, deps wiring) in one place.
- Reuse the same invoke path from terminal and backend without changing logic.
"""

from __future__ import annotations

import json
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Iterator

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver

from app.config.settings import Settings, load_settings
from app.deps import AppDeps, clear_stream_emitter, set_deps, set_stream_emitter
from app.graphs.main_graph import build_graph
from app.models.state import ChatMessage, GraphState
from app.persistence.checkpointer import close_checkpointer, open_checkpointer
from app.utils.helpers import build_retriever


@dataclass
class GraphRuntime:
    settings: Settings
    app_graph: object
    checkpointer: PostgresSaver
    _checkpointer_cm: AbstractContextManager[PostgresSaver]


def create_runtime() -> GraphRuntime:
    """Initialize everything once and return a reusable runtime object."""
    settings = load_settings()
    retriever = build_retriever(settings)
    llm = ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)
    set_deps(AppDeps(llm=llm, retriever=retriever, settings=settings))

    checkpointer, checkpointer_cm = open_checkpointer(settings.database_url)
    app_graph = build_graph(settings, llm, retriever, checkpointer=checkpointer)
    return GraphRuntime(
        settings=settings,
        app_graph=app_graph,
        checkpointer=checkpointer,
        _checkpointer_cm=checkpointer_cm,
    )


def close_runtime(runtime: GraphRuntime) -> None:
    """Release the Postgres checkpointer connection."""
    close_checkpointer(runtime._checkpointer_cm)


def new_thread_id() -> str:
    """Generate a unique LangGraph thread id for a new chat session."""
    return str(uuid.uuid4())


def build_invoke_config(runtime: GraphRuntime, thread_id: str) -> dict[str, Any]:
    """Runnable config: recursion limit + Postgres thread id."""
    return {
        "recursion_limit": runtime.settings.graph_recursion_limit,
        "configurable": {"thread_id": thread_id},
    }


def build_initial_state(question: str) -> GraphState:
    """Build per-turn graph state (does not reset persisted chat_history)."""
    return {
        "question": question,
        # Start retrieval query from the user question; rewrite node can improve it.
        "retrieval_query": question,
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
        "answer_source": "documents",
    }


def _append_chat_turn(
    runtime: GraphRuntime,
    config: dict[str, Any],
    question: str,
    answer: str,
) -> None:
    """Append user + assistant messages to the thread's persisted history."""
    turn: list[ChatMessage] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    runtime.app_graph.update_state(config, {"chat_history": turn})


def ask_question(runtime: GraphRuntime, question: str, thread_id: str) -> GraphState:
    """
    Run the self-RAG graph for one user turn.

    ``thread_id`` selects the Postgres-backed checkpoint thread (one per chat).
    """
    config = build_invoke_config(runtime, thread_id)
    initial_state = build_initial_state(question)
    result: GraphState = runtime.app_graph.invoke(initial_state, config=config)

    answer = str(result.get("answer", "")).strip()
    if answer:
        _append_chat_turn(runtime, config, question, answer)

    return result


def get_chat_history(runtime: GraphRuntime, thread_id: str) -> list[ChatMessage]:
    """Return persisted messages for a thread (empty if the chat does not exist yet)."""
    config = build_invoke_config(runtime, thread_id)
    snapshot = runtime.app_graph.get_state(config)
    if snapshot is None or not snapshot.values:
        return []
    return list(snapshot.values.get("chat_history") or [])


STATUS_LABELS: dict[str, str] = {
    "decide_answer_source": "deciding",
    "generate_from_history": "recalling",
    "decide_retrieval": "deciding",
    "retrieve": "retrieving",
    "is_relevant": "grading",
    "generate_from_context": "generating",
    "generate_direct": "generating",
    "is_sup": "verifying",
    "revise_answer": "revising",
    "is_use": "checking",
    "rewrite_question": "rewriting",
    "no_answer_found": "finishing",
}


def format_sse(event_type: str, data: dict[str, Any] | None = None) -> str:
    """Format one Server-Sent Event line block."""
    payload: dict[str, Any] = {"type": event_type}
    if data:
        payload.update(data)
    return f"data: {json.dumps(payload)}\n\n"


def graph_result_to_done_payload(thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Build the ``done`` event fields from graph state."""
    return {
        "thread_id": thread_id,
        "answer": str(result.get("answer", "")),
        "need_retrieval": result.get("need_retrieval"),
        "issup": result.get("issup"),
        "evidence": result.get("evidence", []) or [],
        "isuse": result.get("isuse"),
        "use_reason": result.get("use_reason"),
        "rewrite_tries": result.get("rewrite_tries", 0) or 0,
        "retries": result.get("retries", 0) or 0,
    }


def ask_question_stream(
    runtime: GraphRuntime,
    question: str,
    thread_id: str,
) -> Iterator[str]:
    """
    Run the self-RAG graph and yield SSE chunks (thread_id, status, token, done, error).

    Uses synchronous ``graph.stream`` because the Postgres checkpointer does not
  support async ``astream_events``.
    """
    config = build_invoke_config(runtime, thread_id)
    initial_state = build_initial_state(question)
    pending: list[str] = []

    def on_token(content: str) -> None:
        pending.append(format_sse("token", {"content": content}))

    def on_clear() -> None:
        pending.append(format_sse("clear"))

    set_stream_emitter(on_token, on_clear)

    yield format_sse("thread_id", {"thread_id": thread_id})

    try:
        for chunk in runtime.app_graph.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                if node_name in STATUS_LABELS:
                    pending.append(format_sse("status", {"label": STATUS_LABELS[node_name]}))

            while pending:
                yield pending.pop(0)

        while pending:
            yield pending.pop(0)

        snapshot = runtime.app_graph.get_state(config)
        values = dict(snapshot.values) if snapshot and snapshot.values else {}

        answer = str(values.get("answer", "")).strip()
        if answer:
            _append_chat_turn(runtime, config, question, answer)

        yield format_sse("done", graph_result_to_done_payload(thread_id, values))
    except Exception as exc:
        yield format_sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
    finally:
        clear_stream_emitter()
