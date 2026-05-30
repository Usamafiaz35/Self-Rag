# Self-RAG Chat UI

Standalone ChatGPT-style frontend for the Self-RAG backend. This folder can be moved to its own repository later.

## Prerequisites

- Backend running with valid `.env` (`OPENAI_API_KEY`, `DATABASE_URL`)
- PDFs in `documents/` (see project root README)

## Run

**Terminal 1 — API** (from project root):

```bash
python scripts/run_api.py
```

From project root, or run `run_api.bat` on Windows. This stops any stale process on port 8000 before starting (avoids `WinError 10013`).

**Terminal 2 — frontend** (from this folder):

```bash
python -m http.server 5500
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500).

## Configuration

Edit [`js/config.js`](js/config.js) if the API is not on `http://127.0.0.1:8000`:

```javascript
export const API_BASE_URL = "http://127.0.0.1:8000";
```

CORS on the backend allows origins on ports `5500` and `5173`.

## Features

- New chat and recent threads in the sidebar
- Load previous conversations from Postgres-backed checkpoints
- Streaming answers via `POST /ask/stream` (SSE)
