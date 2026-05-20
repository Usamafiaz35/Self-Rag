"""
Thin helpers for joining context and running the RAG generation prompt.

Why this file exists: mirrors the notebook's ``rag_generation_prompt`` + ``llm.invoke`` path
in one place so ``generate_from_context`` stays small and prompts stay in ``prompts/``.
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.prompts.prompts import rag_generation_prompt


def join_context(relevant_docs: Iterable[Document]) -> str:
    """Same joining as the notebook: chunks separated by blank lines and ``---``."""
    return "\n\n---\n\n".join(d.page_content for d in relevant_docs).strip()


def invoke_rag_answer(llm: ChatOpenAI, question: str, context: str) -> BaseMessage:
    """Single LLM call used after context is built (same as ``generate_from_context``)."""
    return llm.invoke(rag_generation_prompt.format_messages(question=question, context=context))
