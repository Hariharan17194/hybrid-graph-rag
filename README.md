# Hybrid RAG (Vector + Graph) — Local App

A local Retrieval-Augmented Generation app that answers questions from
uploaded PDFs using **two** retrieval paths combined:

- **Vector RAG** — ChromaDB similarity search over chunked, embedded text.
- **Graph RAG** — a NetworkX knowledge graph, built by asking the LLM to
  extract entities/relationships per chunk, then traversing 1-hop
  neighborhoods of any entity mentioned in the question.

Both sets of matching chunks are merged and passed to the LLM as labeled
context (`[V1]`, `[G1]`, ...), so answers can cite exactly where they came
from. Input/output **guardrails** screen for empty/oversized/injection-style
questions before they reach the LLM, and flag ungrounded or secret-leaking
answers before they reach you.

```
├── backend/
│   ├── main.py            # FastAPI app: /upload, /query, /status, /health
│   ├── rag_pipeline.py     # chunking, embedding, graph extraction, retrieval
│   ├── guardrails.py       # input/output safety checks
│   ├── config.py           # settings loaded from .env
│   ├── llm_providers.py    # Ollama / OpenAI / Anthropic + embeddings switch
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app.py               # Streamlit chat UI
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## 1. Configure

```bash
cp backend/.env.example backend/.env
```

The defaults run **fully locally, with no API keys**:
- LLM via **Ollama** (`llama3.1`) — install from [ollama.com](https://ollama.com),
  then `ollama pull llama3.1` and make sure `ollama serve` is running.
- Embeddings via a local **sentence-transformers** model (downloaded
  automatically on first use).

To use OpenAI or Anthropic instead, edit `backend/.env`:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```
(embeddings can stay `local`, or set `EMBEDDING_PROVIDER=openai` with an
`OPENAI_API_KEY`.)

## 2. Run

```bash
docker compose up --build
```

- Frontend (Streamlit): http://localhost:8501
- Backend (FastAPI docs): http://localhost:8000/docs

If you're using Ollama, run it on your host machine (not inside Docker) —
the backend container reaches it via `OLLAMA_BASE_URL` in `.env`. On
Linux, `http://localhost:11434` may need to become
`http://host.docker.internal:11434` depending on your Docker setup.

## 3. Use it

1. Open the Streamlit UI, upload a PDF from the sidebar, click **Ingest
   document**. You'll see chunk/entity/relationship counts once it's done.
2. Ask a question in the chat box. Expand **Sources** under any answer to
   see which chunks (vector- or graph-retrieved) it drew from.

## Running without Docker (for development in VS Code)

```bash
# Terminal 1
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- **Persistence**: the vector store, graph, and a manifest of ingested
  files live under `backend/storage/` (mounted as a volume in Compose), so
  your knowledge base survives container restarts.
- **Currently PDF-only**. Extending to other formats means adding an
  extractor in `rag_pipeline._extract_pdf_text`-style function and routing
  by file extension in `main.upload_document`.
- **Deploying behind Traefik**: `docker-compose.yml` has commented-out
  labels for both services — uncomment them, set your hostnames, and point
  them at your existing Traefik network instead of publishing the `ports`
  directly.
- **Guardrails** here are intentionally simple regex/heuristic checks
  (see `backend/guardrails.py`) meant to show the pattern. Swap in a
  library like `guardrails-ai` for production-grade validation.
