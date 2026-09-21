"""
FastAPI backend for the Hybrid (Vector + Graph) RAG application.

Endpoints
---------
POST /upload   -> ingest a PDF: extract text, chunk it, embed it into the
                   vector store, and extract entities/relationships into
                   the graph store.
POST /query    -> answer a question using hybrid retrieval (vector + graph)
                   with input/output guardrails applied.
GET  /status   -> current knowledge base stats (docs, chunks, entities...).
GET  /health   -> simple liveness check.

Run directly with:  uvicorn main:app --reload
Or via Docker:       docker compose up --build
"""

import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from guardrails import check_input, check_output
from rag_pipeline import HybridRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid-rag-backend")

app = FastAPI(title="Hybrid RAG Backend", version="1.0.0")

# Allow the Streamlit frontend to call this API. Restrict allow_origins in
# production if the API is ever exposed beyond your own frontend container.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A single pipeline instance holds the vector store + graph for the process
# lifetime. State is persisted to disk (storage/), so it survives restarts
# and is visible to any worker process.
pipeline = HybridRAGPipeline(persist_dir="storage")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question")
    top_k: int = Field(5, ge=1, le=20, description="Chunks to retrieve per source")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    guardrail_notice: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return pipeline.get_stats()


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    dest = UPLOAD_DIR / file.filename
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        dest.write_bytes(contents)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to save upload")
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}") from exc

    try:
        start = time.time()
        result = pipeline.ingest_pdf(dest)
        elapsed = round(time.time() - start, 2)
        logger.info("Ingested %s in %ss", file.filename, elapsed)
        return {
            "filename": file.filename,
            "chunks": result["chunks"],
            "entities": result["entities"],
            "relationships": result["relationships"],
            "seconds": elapsed,
        }
    except ValueError as exc:
        # Raised by rag_pipeline for malformed / unreadable / empty PDFs.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    guard = check_input(req.question)
    if not guard.passed:
        raise HTTPException(status_code=400, detail=guard.reason)

    if pipeline.get_stats()["documents"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Upload a PDF first.",
        )

    try:
        result = pipeline.answer(req.question, top_k=req.top_k)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="The language model timed out. Please try again."
        ) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    out_guard = check_output(answer=result["answer"], context_chunks=result["sources"])
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        guardrail_notice=None if out_guard.passed else out_guard.reason,
    )
