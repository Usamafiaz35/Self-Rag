"""
One-time (or when PDFs change) index build for production.

Chunks company PDFs, embeds them, and saves FAISS vectors under data/vector_index/.

Usage (from project root):

    python scripts/build_vector_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python scripts/build_vector_index.py`` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import load_settings
from app.utils.vector_store import build_and_save_index, index_exists, vector_index_dir


def main() -> None:
    settings = load_settings()
    out = vector_index_dir(settings)
    print("Building vector index from company PDFs...")
    print(f"  PDFs: {len(settings.pdf_paths)} file(s)")
    print(f"  Chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")
    print(f"  Embedding model: {settings.embedding_model}")

    saved = build_and_save_index(settings)
    print(f"\nDone. Index saved to: {saved}")
    if index_exists(settings):
        print("  Files: index.faiss, index.pkl")
    print("\nStart the chatbot with: python -m uvicorn app.api.main:app --reload")


if __name__ == "__main__":
    main()
