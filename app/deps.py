"""
Runtime dependencies (LLM, retriever, settings) shared by graph nodes.

Nodes read these via get_deps() after run.py calls set_deps() at startup.
This avoids threading many arguments through every node factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from app.config.settings import Settings


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
