"""
Retrieval routing: decide_retrieval plus conditional-edge routers.

Router return strings match the ``path_map`` keys in ``main_graph.py`` (see README).
"""

from __future__ import annotations

from typing import Literal

from app.deps import get_deps
from app.models.state import GraphState, RetrieveDecision
from app.prompts.prompts import decide_retrieval_prompt
from app.utils.chat_history import history_from_state


def decide_retrieval(state: GraphState):
    deps = get_deps()
    should_retrieve_llm = deps.llm.with_structured_output(RetrieveDecision)
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(
            question=state["question"],
            chat_history=history_from_state(state),
        )
    )
    return {"need_retrieval": decision.should_retrieve}


def route_after_decide(state: GraphState) -> Literal["True", "False"]:
    """Match keys in ``add_conditional_edges(..., {"False": ..., "True": ...})``."""
    return "True" if state["need_retrieval"] else "False"


def route_after_relevance(state: GraphState) -> Literal["1 doc found", "no relevant docs"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "1 doc found"
    return "no relevant docs"


def route_after_issup(state: GraphState) -> Literal["fully_supported", "partially_supported with limit 10"]:
    """Match keys after ``is_sup``: proceed to ``is_use`` vs ``revise_answer``."""
    settings = get_deps().settings
    if state.get("issup") == "fully_supported":
        return "fully_supported"
    if state.get("retries", 0) >= settings.max_retries:
        return "fully_supported"
    return "partially_supported with limit 10"


def route_after_isuse(state: GraphState) -> Literal["useful", "not_useful", "rewrite_tries max 3"]:
    settings = get_deps().settings
    if state.get("isuse") == "useful":
        return "useful"
    if state.get("rewrite_tries", 0) >= settings.max_rewrite_tries:
        return "rewrite_tries max 3"
    return "not_useful"
