"""
Generation and verification nodes: direct answer, RAG generate, IsSUP/IsUSE, revise, rewrite.

Same logic as the notebook cells; ``accept_answer`` is kept as in the notebook (unused by the graph).
"""

from __future__ import annotations

from app.chains.rag_chain import invoke_rag_answer, join_context
from app.deps import emit_stream_clear, get_deps, stream_llm_text
from app.models.state import (
    GraphState,
    IsSUPDecision,
    IsUSEDecision,
    RewriteDecision,
)
from app.prompts.prompts import (
    direct_generation_prompt,
    issup_prompt,
    isuse_prompt,
    revise_prompt,
    rewrite_for_retrieval_prompt,
)
from app.utils.chat_history import history_from_state


def generate_direct(state: GraphState):
    deps = get_deps()
    messages = direct_generation_prompt.format_messages(
        question=state["question"],
        chat_history=history_from_state(state),
    )
    return {"answer": stream_llm_text(deps.llm, messages)}


def generate_from_context(state: GraphState):
    deps = get_deps()
    context = join_context(state.get("relevant_docs", []))
    if not context:
        return {"answer": "No answer found.", "context": ""}
    answer = invoke_rag_answer(
        deps.llm,
        state["question"],
        context,
        chat_history=history_from_state(state),
    )
    return {"answer": answer, "context": context}


def no_answer_found(state: GraphState):
    return {"answer": "No answer found.", "context": ""}


def is_sup(state: GraphState):
    deps = get_deps()
    issup_llm = deps.llm.with_structured_output(IsSUPDecision)
    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
            chat_history=history_from_state(state),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


def accept_answer(state: GraphState):
    return {}  # keep answer as-is (notebook defined but not added as a graph node)


def revise_answer(state: GraphState):
    deps = get_deps()
    emit_stream_clear()
    messages = revise_prompt.format_messages(
        question=state["question"],
        answer=state.get("answer", ""),
        context=state.get("context", ""),
    )
    return {
        "answer": stream_llm_text(deps.llm, messages),
        "retries": state.get("retries", 0) + 1,
    }


def is_use(state: GraphState):
    deps = get_deps()
    isuse_llm = deps.llm.with_structured_output(IsUSEDecision)
    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            chat_history=history_from_state(state),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}


def rewrite_question(state: GraphState):
    deps = get_deps()
    rewrite_llm = deps.llm.with_structured_output(RewriteDecision)
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }
