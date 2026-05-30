"""
Compile the LangGraph ``StateGraph`` (same topology as notebook cell 17).

Only wiring lives here—node bodies are in ``app.nodes``.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.config.settings import Settings
from app.models.state import GraphState
from app.nodes.conversation import (
    decide_answer_source,
    generate_from_history,
    route_after_answer_source,
)
from app.nodes.generate import (
    generate_direct,
    generate_from_context,
    is_sup,
    is_use,
    no_answer_found,
    revise_answer,
    rewrite_question,
)
from app.nodes.grade_documents import is_relevant
from app.nodes.retrieve import retrieve
from app.nodes.route_question import (
    decide_retrieval,
    route_after_decide,
    route_after_isuse,
    route_after_issup,
    route_after_relevance,
)


def build_graph(
    settings: Settings,
    llm: ChatOpenAI,
    retriever: BaseRetriever,
    checkpointer: BaseCheckpointSaver,
):
    """
    Build and compile the graph.

    ``settings``, ``llm``, and ``retriever`` are listed so callers document what they
    wired into ``set_deps()``; nodes read the live objects via ``get_deps()``.
    """
    _ = settings, llm, retriever

    g = StateGraph(GraphState)

    g.add_node("decide_answer_source", decide_answer_source)
    g.add_node("generate_from_history", generate_from_history)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)

    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)

    g.add_node("is_sup", is_sup)
    g.add_node("revise_answer", revise_answer)

    g.add_node("is_use", is_use)

    g.add_node("rewrite_question", rewrite_question)

    g.add_edge(START, "decide_answer_source")

    g.add_conditional_edges(
        "decide_answer_source",
        route_after_answer_source,
        {"conversation": "generate_from_history", "documents": "decide_retrieval"},
    )

    g.add_edge("generate_from_history", END)

    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"False": "generate_direct", "True": "retrieve"},
    )

    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "is_relevant")

    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "1 doc found": "generate_from_context",
            "no relevant docs": "no_answer_found",
        },
    )

    g.add_edge("no_answer_found", END)

    g.add_edge("generate_from_context", "is_sup")

    g.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {
            "fully_supported": "is_use",
            "partially_supported with limit 10": "revise_answer",
        },
    )

    g.add_edge("revise_answer", "is_sup")

    g.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {
            "useful": END,
            "not_useful": "rewrite_question",
            "rewrite_tries max 3": "no_answer_found",
        },
    )

    g.add_edge("rewrite_question", "retrieve")

    return g.compile(checkpointer=checkpointer)
