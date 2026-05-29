"""
FastAPI backend for the existing self-RAG graph.

Run with:
    uvicorn app.api.main:app --reload
Then open:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.persistence.checkpointer import list_thread_ids
from app.runtime import (
    GraphRuntime,
    ask_question,
    close_runtime,
    create_runtime,
    get_chat_history,
    new_thread_id,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to pass into the graph.")
    thread_id: str | None = Field(
        default=None,
        description="Chat session id. Omit to start a new chat (a new thread_id is returned).",
    )


class AskResponse(BaseModel):
    thread_id: str
    answer: str
    need_retrieval: bool | None = None
    issup: str | None = None
    evidence: list[str] = Field(default_factory=list)
    isuse: str | None = None
    use_reason: str | None = None
    rewrite_tries: int = 0
    retries: int = 0


class CreateChatResponse(BaseModel):
    thread_id: str = Field(..., description="Unique id for this chat session.")


class ChatMessageOut(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessageOut] = Field(default_factory=list)


class ChatListResponse(BaseModel):
    thread_ids: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = create_runtime()
    app.state.runtime = runtime
    try:
        yield
    finally:
        close_runtime(runtime)


app = FastAPI(
    title="Self-RAG Backend",
    description="FastAPI wrapper around the existing LangGraph self-RAG workflow.",
    version="1.0.0",
    lifespan=lifespan,
)


def _runtime_from_app(fastapi_app: FastAPI) -> GraphRuntime:
    runtime = getattr(fastapi_app.state, "runtime", None)
    if runtime is None:
        # Fallback for contexts that bypass lifespan (some tests/scripts).
        runtime = create_runtime()
        fastapi_app.state.runtime = runtime
    return runtime


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello, main api page!"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chats", response_model=CreateChatResponse)
def create_chat() -> CreateChatResponse:
    """Start a new chat session with a unique thread_id."""
    return CreateChatResponse(thread_id=new_thread_id())


@app.get("/chats", response_model=ChatListResponse)
def list_chats() -> ChatListResponse:
    """List thread ids that have at least one checkpoint in Postgres."""
    runtime = _runtime_from_app(app)
    return ChatListResponse(thread_ids=list_thread_ids(runtime.settings.database_url))


@app.get("/chats/{thread_id}", response_model=ChatHistoryResponse)
def chat_history(thread_id: str) -> ChatHistoryResponse:
    """Return persisted message history for a chat thread."""
    runtime = _runtime_from_app(app)
    messages = get_chat_history(runtime, thread_id)
    return ChatHistoryResponse(
        thread_id=thread_id,
        messages=[ChatMessageOut(role=m["role"], content=m["content"]) for m in messages],
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    thread_id = (payload.thread_id or "").strip() or new_thread_id()
    runtime = _runtime_from_app(app)

    try:
        result: dict[str, Any] = ask_question(runtime, question, thread_id)
    except Exception as exc:  # pragma: no cover - keep response stable for API callers
        raise HTTPException(status_code=500, detail=f"Graph invocation failed: {exc}") from exc

    return AskResponse(
        thread_id=thread_id,
        answer=str(result.get("answer", "")),
        need_retrieval=result.get("need_retrieval"),
        issup=result.get("issup"),
        evidence=result.get("evidence", []) or [],
        isuse=result.get("isuse"),
        use_reason=result.get("use_reason"),
        rewrite_tries=result.get("rewrite_tries", 0) or 0,
        retries=result.get("retries", 0) or 0,
    )
