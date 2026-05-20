# Self-RAG prototype (modular)

LangGraph **self-RAG** workflow refactored from [`notebooks/original_notebook.ipynb`](notebooks/original_notebook.ipynb) (copy of `self_rag_step7.ipynb`). Same node logic and graph wiring as the notebook; optional `print`/diagram cells were not ported.

## Layout

| Path | Why it exists |
|------|----------------|
| [`app/config/settings.py`](app/config/settings.py) | Single place for PDF paths, models, chunking, `k`, retry limits. |
| [`app/models/state.py`](app/models/state.py) | `GraphState` + Pydantic decision schemas. |
| [`app/prompts/prompts.py`](app/prompts/prompts.py) | All `ChatPromptTemplate` definitions. |
| [`app/chains/rag_chain.py`](app/chains/rag_chain.py) | Join context + one RAG LLM call (used by generate). |
| [`app/utils/helpers.py`](app/utils/helpers.py) | Build FAISS retriever; format the run report string. |
| [`app/nodes/`](app/nodes/) | One module per concern: retrieve, grade, generate, route. |
| [`app/graphs/main_graph.py`](app/graphs/main_graph.py) | Only `StateGraph` construction and edges. |
| [`app/deps.py`](app/deps.py) | Shared `llm`, `retriever`, `settings` after `run.py` initializes. |
| [`app/runtime.py`](app/runtime.py) | Shared bootstrap + invoke helpers used by CLI and FastAPI. |
| [`app/api/main.py`](app/api/main.py) | FastAPI app exposing backend endpoints over the same graph logic. |
| [`run.py`](run.py) | Terminal interactive chat loop using the same runtime/invoke path. |

## Setup

1. Python 3.11+ recommended (matches the notebook metadata).
2. Create a `.env` file in the project root with:

   ```bash
   OPENAI_API_KEY=sk-...
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **FAISS:** `langchain-community` uses FAISS; on most platforms you need `faiss-cpu` (included in `requirements.txt`). If install fails on your OS, see [FAISS install docs](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss).

4. Place PDFs under `documents/`:

   - `Company_Policies.pdf`
   - `Company_Profile.pdf`
   - `Product_and_Pricing.pdf`

## Run

From the project root:

```bash
python run.py
```

This starts a terminal chat loop. Type questions and press Enter. Type `exit` to quit.

## FastAPI backend

Start the API server from project root:

```bash
uvicorn app.api.main:app --reload
```

Open FastAPI docs:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Endpoints

- `GET /health` - simple health check.
- `POST /ask` - ask one question using the existing graph.

Example request body for `/ask`:

```json
{
  "question": "Describe NexaAI's company culture."
}
```

Response includes:
- `answer`
- `need_retrieval`
- `issup`, `evidence`
- `isuse`, `use_reason`
- `rewrite_tries`, `retries`

### Conditional routing vs the notebook

The notebook’s `add_conditional_edges` `path_map` keys (e.g. `"False"` / `"True"` for retrieval routing) must match what each router function returns. In the saved notebook, several routers returned **different** strings than those keys; the modular code returns strings that match the **existing** `path_map` destinations (no new branches—only aligned return labels). See [`app/nodes/route_question.py`](app/nodes/route_question.py).

## Future work (out of scope here)

FastAPI, databases, frontend, deployment, and monitoring can be added later without changing the core `app/` graph.


## Created by Usama Fiaz