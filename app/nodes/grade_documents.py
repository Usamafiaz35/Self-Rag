"""Per-document relevance grading after retrieval (notebook ``is_relevant``)."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.deps import get_deps
from app.models.state import GraphState, RelevanceDecision
from app.prompts.prompts import is_relevant_prompt


def is_relevant(state: GraphState):
    deps = get_deps()
    relevance_llm = deps.llm.with_structured_output(RelevanceDecision)
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}
