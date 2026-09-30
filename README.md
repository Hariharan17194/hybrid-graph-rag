<div align="center">

<img src="docs/assets/banner.png" alt="Hybrid Graph RAG" width="100%"/>

# Hybrid Graph RAG

**Ask questions of your PDFs and get answers that cite exactly where they came from — using vector search *and* a knowledge graph.**

[![CI](https://github.com/Hariharan17194/hybrid-graph-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Hariharan17194/hybrid-graph-rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Hariharan17194/hybrid-graph-rag?style=flat-square&color=42E6D4&labelColor=080B10)](https://github.com/Hariharan17194/hybrid-graph-rag/releases)
![Python](https://img.shields.io/badge/python-3.11-42E6D4?style=flat-square&logo=python&logoColor=white&labelColor=080B10)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-42E6D4?style=flat-square&logo=fastapi&logoColor=white&labelColor=080B10)
![Docker](https://img.shields.io/badge/docker-compose-42E6D4?style=flat-square&logo=docker&logoColor=white&labelColor=080B10)
[![License: MIT](https://img.shields.io/badge/license-MIT-FFB454?style=flat-square&labelColor=080B10)](LICENSE)

[**Live demo**](https://tinyurl.com/f2hr8xea) · [**Architecture**](#-architecture) · [**Quick start**](#-quick-start) · [API docs](#-api)

</div>

---

## ✦ Why this exists

Plain vector RAG finds passages that *sound like* your question, but it misses facts spread across a document — "Which supplier provides the part used in product X?" needs two hops. This app builds a **knowledge graph** from the same PDFs and merges graph-retrieved context with vector hits, so answers can connect facts **and** show their evidence.

It runs **fully locally by default** (Ollama + sentence-transformers, no API keys) and switches to OpenAI or Claude with one environment variable.

## ✦ Demo

🔗 **[Try the live app](https://tinyurl.com/f2hr8xea)** (self-hosted on a VPS behind Traefik)

| Cited answer with expandable sources | Prompt-injection guardrail in action |
|---|---|
| ![Answer citing [V1] with an expanded Sources panel](docs/assets/answer.png) | ![Injection attempts blocked with a warning](docs/assets/guardrails.png) |

## ✦ Features

- **Two retrieval paths, one answer** — ChromaDB similarity search + 1-hop NetworkX graph traversal, fused and de-duplicated.
- **Every claim is traceable** — context is labeled `[V1]…` (vector) and `[G1]…` (graph); the model cites them inline and the UI lets you expand each source.
- **LLM-built knowledge graph** — entities and relationships are extracted per chunk at ingest time.
- **Guardrails in and out** — blocks empty/oversized/prompt-injection questions; flags ungrounded or secret-leaking answers.
- **Provider-agnostic** — Ollama (default), OpenAI, or Anthropic for chat; local or OpenAI embeddings.
- **Persistent knowledge base** — vectors, graph and manifest survive container restarts.
- **Deploy-ready** — Docker Compose with health checks and optional Traefik labels for a VPS.

## ✦ Skills demonstrated

**RAG engineering** · embeddings · vector search · knowledge graphs · hybrid retrieval · source-grounded generation · prompt-injection guardrails · FastAPI · Streamlit · Docker · CI/CD

**Course progression:** extends the standard RAG pipeline into the Week 3 knowledge-graph assignment by combining vector retrieval with entity/relation traversal, while also applying Week 2 production concerns such as guardrails and provider abstraction.

## ✦ Architecture

```mermaid
flowchart TB
    subgraph Ingest["📥 Ingest  (POST /upload)"]
        PDF[/PDF/] --> TXT[Extract text · pypdf]
        TXT --> CH[Chunk · 800 chars / 120 overlap]
        CH --> EMB[Embed]
        EMB --> VDB[(ChromaDB)]
        CH --> EXT[LLM entity & relation extraction]
        EXT --> KG[(NetworkX graph)]
    end

    subgraph Query["💬 Query  (POST /query)"]
        Q[/Question/] --> GIN{Input guardrails}
        GIN -- blocked --> R400[400 · reason]
        GIN -- ok --> VS[Vector search · top-k]
        GIN -- ok --> GS[Graph search · entities in question → 1-hop]
        VS --> FUSE[Fuse & de-dup · label V1..Vn, G1..Gn]
        GS --> FUSE
        FUSE --> LLM[LLM answer with inline citations]
        LLM --> GOUT{Output guardrails}
        GOUT --> ANS[/Answer + sources + notice/]
    end

    VDB -.-> VS
    KG -.-> GS

    classDef store fill:#0C1017,stroke:#FFB454,color:#E9EDF2;
    classDef step fill:#0C1017,stroke:#42E6D4,color:#E9EDF2;
    class VDB,KG store;
    classDef io fill:#080B10,stroke:#7E8A99,color:#E9EDF2;
    class TXT,CH,EMB,EXT,VS,GS,FUSE,LLM step;
    class PDF,Q,GIN,GOUT,R400,ANS io;
    style Ingest fill:#080B10,stroke:#1B2330,color:#42E6D4
    style Query fill:#080B10,stroke:#1B2330,color:#42E6D4
```

**Services**

```mermaid
flowchart LR
    B([Browser]) -->|:8501| FE[Streamlit UI]
    FE -->|HTTP| BE[FastAPI backend :8000]
    BE --> ST[(./backend/storage volume)]
    BE -->|OLLAMA_BASE_URL| OL[Ollama on host]
    BE -.->|optional| OA[OpenAI / Anthropic]
```

## ✦ Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Chat + sidebar stats in pure Python |
| API | FastAPI + Pydantic | Typed request/response models, auto OpenAPI docs |
| Vector store | ChromaDB (persistent) | Zero-ops, embedded, good enough for single-node |
| Graph | NetworkX (pickled) | Simple, inspectable; Neo4j is the planned upgrade |
| LLM | Ollama `llama3.1` · OpenAI · Claude | Local-first, swap with one env var |
| Embeddings | `all-MiniLM-L6-v2` · `text-embedding-3-small` | Fast local default, hosted option |
| Packaging | Docker Compose, Traefik labels | One command locally, reverse-proxy ready on a VPS |

## ✦ Quick start

**Prerequisites:** Docker 24+, and either [Ollama](https://ollama.com) running on the host (`ollama pull llama3.1`) or an OpenAI/Anthropic API key.

```bash
git clone https://github.com/Hariharan17194/hybrid-graph-rag.git
cd hybrid-graph-rag
cp backend/.env.example backend/.env
docker compose up --build
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |

> **Ollama + Docker on Linux:** set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `backend/.env`.

<details>
<summary><b>Local development without Docker (VS Code)</b></summary>

```bash
# Terminal 1 — backend
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2 — frontend
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

</details>

## ✦ Configuration

All settings live in `backend/.env` (see [`.env.example`](backend/.env.example)).

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` · `openai` · `anthropic` |
| `OLLAMA_MODEL` | `llama3.1` | Any model you've pulled |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) · `openai` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Characters per chunk / overlap |
| `MAX_QUESTION_CHARS` | `2000` | Input guardrail limit |
| `BLOCK_SUSPECTED_INJECTION` | `true` | Reject prompt-injection patterns |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-call timeout |

## ✦ API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/upload` | Ingest a PDF → `{chunks, entities, relationships, seconds}` |
| `POST` | `/query` | `{question, top_k}` → `{answer, sources[], guardrail_notice}` |
| `GET` | `/status` | Knowledge-base stats |
| `GET` | `/health` | Liveness probe |

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/upload
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What is the refund window for damaged items?", "top_k": 5}'
```

## ✦ Project structure

```text
hybrid-graph-rag/
├── backend/
│   ├── main.py            # FastAPI app: /upload, /query, /status, /health
│   ├── rag_pipeline.py    # chunking, embedding, graph extraction, hybrid retrieval
│   ├── guardrails.py      # input/output safety checks
│   ├── llm_providers.py   # Ollama / OpenAI / Anthropic + embeddings switch
│   ├── config.py          # settings from .env
│   └── tests/             # pytest suite (guardrails, API contract)
├── frontend/app.py        # Streamlit chat UI
├── docs/assets/           # banner + screenshots
└── docker-compose.yml
```

## ✦ Design decisions & trade-offs

- **Graph built by the LLM at ingest time** — richer retrieval, but ingest is slower and costs one LLM call per chunk. A cache keyed by chunk hash is on the roadmap.
- **1-hop traversal only** — keeps context small and relevant; multi-hop reasoning is left to the LLM.
- **Heuristic guardrails** — regex checks show the pattern with zero dependencies; the `check_input` / `check_output` interface is designed so `guardrails-ai` or NeMo Guardrails can drop in.
- **NetworkX over Neo4j** — no extra service to run. Neo4j becomes worthwhile once graphs outgrow memory or need Cypher queries.
- **Known limitations:** PDF-only ingestion; single-user; no auth on the API.

## ✦ Roadmap

- [x] Vector + graph hybrid retrieval with inline citations
- [x] Docker Compose deployment behind Traefik
- [ ] BM25 keyword retrieval + reciprocal-rank fusion
- [ ] Retrieval evaluation set (RAGAS / LangSmith) with scores in this README
- [ ] Neo4j backend option
- [ ] DOCX / Markdown / web-page ingestion

## ✦ Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Commits follow [Conventional Commits](https://www.conventionalcommits.org/); releases are automated with release-please.

## ✦ License

[MIT](LICENSE) © 2026 Hariharan Padmanabhan

<div align="center"><sub>Built by <a href="https://github.com/Hariharan17194">@Hariharan17194</a> · <a href="https://hariharan17194.github.io/">portfolio</a></sub></div>
