"""
Utilities: build the FAISS retriever and print the same run report as the notebook.

Why this file exists: ingestion (load PDFs → chunk → embed) stays out of ``run.py`` and nodes.
"""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import Settings
from app.models.state import GraphState


def build_retriever(settings: Settings) -> BaseRetriever:
    """Load notebooks' PDFs, chunk, embed with OpenAI, return a FAISS retriever (k from settings)."""
    docs = []
    for path in settings.pdf_paths:
        docs.extend(PyPDFLoader(path).load())
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).split_documents(docs)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})


def format_run_report(initial_state: GraphState, result: GraphState) -> str:
    """Human-readable debug output (same structure as the last notebook cell)."""
    lines: list[str] = []
    lines.append("\n===== RAG EXECUTION RESULT =====\n")
    lines.append("")
    lines.append(f"Question: {initial_state.get('question')}")
    lines.append(f"Need Retrieval: {result.get('need_retrieval')}")
    lines.append("")
    lines.append(f"Rewrite tries (retrieval): {result.get('rewrite_tries', 0)}")
    lines.append(f"Support revise tries: {result.get('retries', 0)}")
    lines.append("")
    lines.append("Retrieval:")
    lines.append(f"  Total retrieved docs: {len(result.get('docs', []) or [])}")
    lines.append(f"  Relevant docs: {len(result.get('relevant_docs', []) or [])}")
    lines.append("")
    relevant_docs = result.get("relevant_docs", []) or []
    if relevant_docs:
        lines.append("Relevant docs (source/page):")
        for i, d in enumerate(relevant_docs, 1):
            src = (d.metadata or {}).get("source", "unknown")
            page = (d.metadata or {}).get("page", None)
            title = (d.metadata or {}).get("title", "")
            extra = f", title={title}" if title else ""
            if page is not None:
                lines.append(f"  {i}. source={src}, page={page}{extra}")
            else:
                lines.append(f"  {i}. source={src}{extra}")
    lines.append("")
    lines.append("Verification (IsSUP):")
    lines.append(f"  issup: {result.get('issup')}")
    evidence = result.get("evidence", []) or []
    if evidence:
        lines.append("  evidence:")
        for e in evidence:
            lines.append(f"   - {e}")
    else:
        lines.append("  evidence: (none)")
    lines.append("")
    lines.append("Usefulness (IsUSE):")
    lines.append(f"  isuse: {result.get('isuse')}")
    lines.append(f"  reason: {result.get('use_reason', '')}")
    lines.append("")
    lines.append("Final Answer:")
    lines.append(str(result.get("answer", "")))
    lines.append("")
    lines.append("===============================\n")
    return "\n".join(lines)
