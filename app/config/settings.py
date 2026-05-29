"""
Central knobs for the prototype: PDF locations, models, chunking, limits.

Why this file exists: keep everything you might tune while learning in one module
instead of scattering literals across nodes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable settings loaded once at process start."""

    pdf_paths: tuple[str, ...]
    chunk_size: int = 600
    chunk_overlap: int = 150
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    retriever_k: int = 4
    max_retries: int = 10  # IsSUP revise loop (matches notebook MAX_RETRIES)
    max_rewrite_tries: int = 3  # IsUSE → rewrite retrieval query
    database_url: str = ""
    graph_recursion_limit: int = 80


def load_settings() -> Settings:
    """Load .env (e.g. OPENAI_API_KEY) and build paths relative to project root."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError(
            "DATABASE_URL is required. Example: "
            "postgresql://postgres:password@localhost:5432/self-rag"
        )

    project_root = Path(__file__).resolve().parent.parent.parent
    docs = project_root / "documents"
    return Settings(
        pdf_paths=(
            str(docs / "Company_Policies.pdf"),
            str(docs / "Company_Profile.pdf"),
            str(docs / "Product_and_Pricing.pdf"),
        ),
        database_url=database_url,
    )
