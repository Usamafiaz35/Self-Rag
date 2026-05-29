"""
PostgreSQL checkpointer lifecycle and thread listing helpers.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver


def open_checkpointer(database_url: str) -> tuple[PostgresSaver, AbstractContextManager[PostgresSaver]]:
    """
    Open a PostgresSaver, run migrations once, and return (saver, context).

    Caller must call ``close_checkpointer(context)`` on shutdown.
    """
    cm = PostgresSaver.from_conn_string(database_url)
    saver = cm.__enter__()
    saver.setup()
    return saver, cm


def close_checkpointer(cm: AbstractContextManager[PostgresSaver]) -> None:
    """Exit the PostgresSaver context manager."""
    cm.__exit__(None, None, None)


def list_thread_ids(database_url: str) -> list[str]:
    """Return distinct LangGraph thread IDs stored in Postgres."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE checkpoint_ns = ''
                ORDER BY thread_id
                """
            )
            rows: list[Any] = cur.fetchall()
    return [str(row[0]) for row in rows]
