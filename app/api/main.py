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

from app.runtime import GraphRuntime, ask_question, create_runtime


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to pass into the graph.")


class AskResponse(BaseModel):
    answer: str
    need_retrieval: bool | None = None
    issup: str | None = None
    evidence: list[str] = Field(default_factory=list)
    isuse: str | None = None
    use_reason: str | None = None
    rewrite_tries: int = 0
    retries: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = create_runtime()
    yield


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result: dict[str, Any] = ask_question(_runtime_from_app(app), question)
    except Exception as exc:  # pragma: no cover - keep response stable for API callers
        raise HTTPException(status_code=500, detail=f"Graph invocation failed: {exc}") from exc

    return AskResponse(
        answer=str(result.get("answer", "")),
        need_retrieval=result.get("need_retrieval"),
        issup=result.get("issup"),
        evidence=result.get("evidence", []) or [],
        isuse=result.get("isuse"),
        use_reason=result.get("use_reason"),
        rewrite_tries=result.get("rewrite_tries", 0) or 0,
        retries=result.get("retries", 0) or 0,
    )
