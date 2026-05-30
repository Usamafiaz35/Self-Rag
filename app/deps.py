"""
Runtime dependencies (LLM, retriever, settings) shared by graph nodes.

Nodes read these via get_deps() after run.py calls set_deps() at startup.
This avoids threading many arguments through every node factory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from langchain_core.messages import BaseMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from app.config.settings import Settings

TokenEmitter = Callable[[str], None]
_stream_local = threading.local()


def set_stream_emitter(
    on_token: TokenEmitter | None,
    on_clear: Callable[[], None] | None = None,
) -> None:
    """Register callbacks used while ``/ask/stream`` runs (thread-local for worker threads)."""
    _stream_local.on_token = on_token
    _stream_local.on_clear = on_clear


def clear_stream_emitter() -> None:
    """Clear stream callbacks after a request finishes."""
    _stream_local.on_token = None
    _stream_local.on_clear = None


def emit_stream_token(content: str) -> None:
    if not content:
        return
    emitter = getattr(_stream_local, "on_token", None)
    if emitter is not None:
        emitter(content)


def emit_stream_clear() -> None:
    clearer = getattr(_stream_local, "on_clear", None)
    if clearer is not None:
        clearer()


def stream_llm_text(llm: ChatOpenAI, messages: list[BaseMessage]) -> str:
    """Invoke or stream the LLM; emit tokens when a stream emitter is registered."""
    emitter = getattr(_stream_local, "on_token", None)
    if emitter is None:
        return str(llm.invoke(messages).content)

    parts: list[str] = []
    for chunk in llm.stream(messages):
        piece = chunk.content
        if not piece:
            continue
        parts.append(piece)
        emitter(piece)
    return "".join(parts)


@dataclass
class AppDeps:
    """Wiring used by LangGraph nodes."""

    llm: ChatOpenAI
    retriever: BaseRetriever
    settings: "Settings"


_deps: AppDeps | None = None


def set_deps(deps: AppDeps) -> None:
    global _deps
    _deps = deps


def get_deps() -> AppDeps:
    if _deps is None:
        raise RuntimeError("Application dependencies not initialized; call set_deps() from run.py first.")
    return _deps
