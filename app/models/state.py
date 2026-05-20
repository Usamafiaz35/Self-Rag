"""
LangGraph state and structured-output schemas.

GraphState mirrors the notebook's TypedDict ``State``.
Pydantic models match the notebook's decision classes for ``with_structured_output``.
"""

from __future__ import annotations

from typing import List, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GraphState(TypedDict):
    question: str

    retrieval_query: str
    rewrite_tries: int

    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str

    # Notebook initial_state used ""; values match IsSUPDecision after ``is_sup``.
    issup: str
    evidence: List[str]

    retries: int

    isuse: Literal["useful", "not_useful"]
    use_reason: str


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer reliably, else False.",
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True ONLY if the document contains info that can directly answer the question.",
    )


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal company PDFs.",
    )
