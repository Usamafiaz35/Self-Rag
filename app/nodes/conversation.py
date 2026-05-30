"""
Answer questions from persisted chat history (conversation memory).
"""

from __future__ import annotations

from typing import Literal

from app.deps import get_deps, stream_llm_text
from app.models.state import AnswerSourceDecision, GraphState
from app.prompts.prompts import answer_source_prompt, history_generation_prompt
from app.utils.chat_history import history_from_state, is_likely_conversation_question


def decide_answer_source(state: GraphState):
    """Route to conversation memory vs company-document RAG."""
    history = list(state.get("chat_history") or [])
    if not history:
        return {"answer_source": "documents"}

    question = state["question"]
    if is_likely_conversation_question(question):
        return {"answer_source": "conversation"}

    deps = get_deps()
    classifier = deps.llm.with_structured_output(AnswerSourceDecision)
    decision: AnswerSourceDecision = classifier.invoke(
        answer_source_prompt.format_messages(
            question=question,
            chat_history=history_from_state(state),
        )
    )
    return {"answer_source": decision.source}


def route_after_answer_source(state: GraphState) -> Literal["conversation", "documents"]:
    if state.get("answer_source") == "conversation":
        return "conversation"
    return "documents"


def generate_from_history(state: GraphState):
    """Answer using only prior user/assistant messages in this thread."""
    deps = get_deps()
    history_text = history_from_state(state)
    messages = history_generation_prompt.format_messages(
        question=state["question"],
        chat_history=history_text,
    )
    return {"answer": stream_llm_text(deps.llm, messages), "context": ""}
