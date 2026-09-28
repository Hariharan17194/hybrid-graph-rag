"""
Hybrid RAG pipeline: combines a vector store (ChromaDB) with a knowledge
graph (NetworkX) built by LLM-based entity/relationship extraction.

Flow
----
Ingest:  PDF -> text -> chunks -> [vector store]  and  -> [LLM extracts
         entities/relationships per chunk] -> [graph store]

Query:   question -> vector similarity search (top_k chunks)
                   + graph lookup (entities mentioned in the question,
                     expanded to their 1-hop neighborhood)
                   -> fused, deduped context
                   -> LLM answer, grounded in that context
"""

import json
import logging
import pickle
import re
import uuid
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx
from pypdf import PdfReader

from config import settings
from llm_providers import LLMError, chat_completion, embed_texts

logger = logging.getLogger("hybrid-rag-backend")

ENTITY_EXTRACTION_PROMPT = """\
Extract the key entities and relationships from the passage below.
Return ONLY valid JSON (no markdown fences, no commentary) in this exact shape:

{{
  "entities": [{{"name": "...", "type": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "relation": "..."}}]
}}

Rules:
- Keep entity names short and consistent (e.g. always "PostgreSQL", not
  "the PostgreSQL database" in one place and "Postgres" in another).
- Only include relationships between entities you listed.
- If nothing meaningful is present, return {{"entities": [], "relationships": []}}.

Passage:
\"\"\"
{chunk}
\"\"\"
"""

ANSWER_PROMPT = """\
You are a helpful assistant answering questions using ONLY the context below.
If the context does not contain the answer, say so plainly instead of guessing.
Cite sources inline using their bracketed labels, e.g. [V1] or [G2].

Context:
{context}

Question: {question}

Answer:
"""


class HybridRAGPipeline:
    def __init__(self, persist_dir: str = "storage"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )

        self.graph_path = self.persist_dir / "graph.pickle"
        self.graph: nx.Graph = self._load_graph()

        self.manifest_path = self.persist_dir / "manifest.json"
        self.manifest: dict[str, Any] = self._load_manifest()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_graph(self) -> nx.Graph:
        if self.graph_path.exists():
            with open(self.graph_path, "rb") as f:
                return pickle.load(f)
        return nx.Graph()

    def _save_graph(self) -> None:
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"documents": []}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def get_stats(self) -> dict[str, int]:
        return {
            "documents": len(self.manifest["documents"]),
            "chunks": self.collection.count(),
            "entities": self.graph.number_of_nodes(),
            "relationships": self.graph.number_of_edges(),
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            raise ValueError(f"Could not open '{pdf_path.name}' as a PDF: {exc}") from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(f"'{pdf_path.name}' is password-protected.") from exc

        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                logger.warning("Failed to extract text from a page in %s", pdf_path.name)

        text = "\n".join(pages).strip()
        if not text:
            raise ValueError(
                f"No extractable text found in '{pdf_path.name}' "
                "(it may be a scanned/image-only PDF)."
            )
        return text

    def _chunk_text(self, text: str) -> list[str]:
        """Simple fixed-size character chunker with overlap. Splits on
        paragraph boundaries where possible so chunks stay readable."""
        size, overlap = settings.chunk_size, settings.chunk_overlap
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 <= size:
                current = f"{current}\n{para}".strip()
            else:
                if current:
                    chunks.append(current)
                # A single paragraph longer than the chunk size gets hard-split.
                while len(para) > size:
                    chunks.append(para[:size])
                    para = para[size - overlap :]
                current = para
        if current:
            chunks.append(current)
        return chunks

    def _extract_graph_from_chunk(self, chunk: str, chunk_id: str) -> tuple[int, int]:
        """Ask the LLM for entities/relationships in this chunk and merge
        them into the graph. Returns (entities_added, relationships_added).
        Failures here are logged and skipped -- graph extraction is a
        best-effort enrichment, not something that should fail ingestion."""
        try:
            raw = chat_completion(
                ENTITY_EXTRACTION_PROMPT.format(chunk=chunk[:3000]),
                system="You extract structured knowledge graphs from text.",
            )
            data = json.loads(_strip_json_fence(raw))
        except (LLMError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Graph extraction skipped for a chunk: %s", exc)
            return 0, 0

        added_entities, added_rels = 0, 0
        for ent in data.get("entities", []):
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            if self.graph.has_node(name):
                self.graph.nodes[name]["chunk_ids"].add(chunk_id)
            else:
                self.graph.add_node(name, type=ent.get("type", "unknown"), chunk_ids={chunk_id})
                added_entities += 1

        for rel in data.get("relationships", []):
            src, tgt = str(rel.get("source", "")).strip(), str(rel.get("target", "")).strip()
            if not src or not tgt or src == tgt:
                continue
            if not self.graph.has_node(src):
                self.graph.add_node(src, type="unknown", chunk_ids={chunk_id})
            if not self.graph.has_node(tgt):
                self.graph.add_node(tgt, type="unknown", chunk_ids={chunk_id})
            if self.graph.has_edge(src, tgt):
                self.graph[src][tgt]["chunk_ids"].add(chunk_id)
            else:
                self.graph.add_edge(
                    src, tgt, relation=rel.get("relation", "related_to"), chunk_ids={chunk_id}
                )
                added_rels += 1
        return added_entities, added_rels

    def ingest_pdf(self, pdf_path: Path) -> dict[str, int]:
        text = self._extract_pdf_text(pdf_path)
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError(f"'{pdf_path.name}' produced no usable text chunks.")

        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = embed_texts(chunks)
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": pdf_path.name, "chunk_index": i} for i in range(len(chunks))],
        )

        total_entities, total_rels = 0, 0
        for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
            e, r = self._extract_graph_from_chunk(chunk, chunk_id)
            total_entities += e
            total_rels += r
        self._save_graph()

        self.manifest["documents"].append({"filename": pdf_path.name, "chunks": len(chunks)})
        self._save_manifest()

        return {"chunks": len(chunks), "entities": total_entities, "relationships": total_rels}

    # ------------------------------------------------------------------
    # Retrieval + answering
    # ------------------------------------------------------------------

    def _vector_search(self, question: str, top_k: int) -> list[dict]:
        query_embedding = embed_texts([question])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits = []
        for doc, meta, chunk_id, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
            results["distances"][0],
            strict=True,
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": 1 - distance,
                    "retrieval": "vector",
                }
            )
        return hits

    def _graph_search(self, question: str, top_k: int) -> list[dict]:
        """Find graph nodes mentioned in the question, pull in their
        1-hop neighborhood, and map back to originating chunks."""
        q_lower = question.lower()
        matched_nodes = [n for n in self.graph.nodes if n.lower() in q_lower]
        if not matched_nodes:
            return []

        related_chunk_ids: set[str] = set()
        for node in matched_nodes:
            related_chunk_ids |= self.graph.nodes[node].get("chunk_ids", set())
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph[node][neighbor]
                related_chunk_ids |= edge_data.get("chunk_ids", set())

        if not related_chunk_ids:
            return []

        fetched = self.collection.get(ids=list(related_chunk_ids)[: top_k * 2])
        hits = []
        for doc, meta, chunk_id in zip(
            fetched["documents"], fetched["metadatas"], fetched["ids"], strict=True
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": None,
                    "retrieval": "graph",
                }
            )
        return hits[:top_k]

    def answer(self, question: str, top_k: int = 5) -> dict:
        vector_hits = self._vector_search(question, top_k)
        graph_hits = self._graph_search(question, top_k)

        seen_ids = set()
        fused: list[dict] = []
        for hit in vector_hits + graph_hits:
            if hit["chunk_id"] in seen_ids:
                continue
            seen_ids.add(hit["chunk_id"])
            fused.append(hit)

        if not fused:
            return {
                "answer": "I don't have any relevant information in the ingested documents to answer that.",
                "sources": [],
            }

        labeled_context = []
        for i, hit in enumerate(fused, start=1):
            label = f"{'V' if hit['retrieval'] == 'vector' else 'G'}{i}"
            hit["label"] = label
            labeled_context.append(f"[{label}] (from {hit['source']}): {hit['text']}")

        prompt = ANSWER_PROMPT.format(context="\n\n".join(labeled_context), question=question)
        answer_text = chat_completion(
            prompt, system="You answer strictly and only from the given context."
        )

        return {
            "answer": answer_text.strip(),
            "sources": [
                {
                    "label": h["label"],
                    "source": h["source"],
                    "retrieval": h["retrieval"],
                    "snippet": h["text"][:300],
                }
                for h in fused
            ],
        }


def _strip_json_fence(raw: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` even when told not to."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()
    return raw
