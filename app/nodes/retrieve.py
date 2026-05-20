"""Vector retrieval node (same behavior as the notebook)."""

from __future__ import annotations

from app.deps import get_deps
from app.models.state import GraphState


def retrieve(state: GraphState):
    retriever = get_deps().retriever
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}
