"""
Format persisted ``chat_history`` for LLM prompts and detect memory-style questions.
"""

from __future__ import annotations

import re

from app.models.state import ChatMessage

_NO_PRIOR = "(No prior messages in this thread.)"

_MEMORY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmy name\b",
        r"\bwhat(?:'s| is) my name\b",
        r"\btell me my name\b",
        r"\bwhat did i ask\b",
        r"\bquestions? i asked\b",
        r"\basked questions?\b",
        r"\bquestions?.*\basked\b",
        r"\bwhat(?:'s| have) i asked\b",
        r"\btell me my\b",
        r"\bprevious questions?\b",
        r"\bearlier (?:i |we )?(?:asked|said|told)\b",
        r"\bi told you\b",
        r"\byou (?:said|told)\b",
        r"\bremember\b",
        r"\bthis (?:chat|conversation)\b",
        r"\bour conversation\b",
        r"\bchat history\b",
        r"\blist (?:all )?(?:my |the )?questions\b",
        r"\bwhat did (?:i|we) (?:say|discuss|talk)\b",
        r"\brecall\b",
    )
)


def format_chat_history(messages: list[ChatMessage] | None) -> str:
    """Render prior turns as plain text for prompt injection."""
    if not messages:
        return _NO_PRIOR

    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else _NO_PRIOR


def history_from_state(state: dict) -> str:
    """Read ``chat_history`` from graph state and format for prompts."""
    return format_chat_history(list(state.get("chat_history") or []))


def is_likely_conversation_question(question: str) -> bool:
    """Fast check for questions about this chat, not company PDFs."""
    text = (question or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _MEMORY_PATTERNS)
