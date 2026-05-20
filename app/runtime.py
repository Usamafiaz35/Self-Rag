"""
Shared runtime bootstrap for both CLI and FastAPI.

Why this file exists:
- Keep initialization (settings, retriever, llm, graph, deps wiring) in one place.
- Reuse the same invoke path from terminal and backend without changing logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from app.config.settings import Settings, load_settings
from app.deps import AppDeps, set_deps
from app.graphs.main_graph import build_graph
from app.models.state import GraphState
from app.utils.helpers import build_retriever


@dataclass
class GraphRuntime:
    settings: Settings
    app_graph: object


def create_runtime() -> GraphRuntime:
    """Initialize everything once and return a reusable runtime object."""
    settings = load_settings()
    retriever = build_retriever(settings)
    llm = ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)
    set_deps(AppDeps(llm=llm, retriever=retriever, settings=settings))
    app_graph = build_graph(settings, llm, retriever)
    return GraphRuntime(settings=settings, app_graph=app_graph)


def build_initial_state(question: str) -> GraphState:
    """Build a fresh graph state for one user question."""
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
    }


def ask_question(runtime: GraphRuntime, question: str) -> GraphState:
    """Invoke the existing graph using the same recursion limit as before."""
    initial_state = build_initial_state(question)
    return runtime.app_graph.invoke(initial_state, config={"recursion_limit": 80})
