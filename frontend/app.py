"""
Streamlit frontend for the Hybrid RAG app.

Talks to the FastAPI backend over HTTP. Run with:
    streamlit run app.py
Or via Docker (see docker-compose.yml).
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # seconds; ingestion/graph extraction can be slow

st.set_page_config(page_title="Hybrid RAG", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def backend_get(path: str):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Can't reach the backend at {BACKEND_URL}. Is it running?"
    except requests.exceptions.Timeout:
        return None, "Backend request timed out."
    except requests.exceptions.HTTPError as exc:
        return None, exc.response.json().get("detail", str(exc))


def backend_post(path: str, **kwargs):
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Can't reach the backend at {BACKEND_URL}. Is it running?"
    except requests.exceptions.Timeout:
        return None, "Request timed out -- the model may be taking a while. Try again."
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except ValueError:
            detail = str(exc)
        return None, detail


if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------------------------------
# Sidebar: upload + knowledge base status
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Ingest document", use_container_width=True):
        with st.spinner(f"Processing '{uploaded_file.name}' (chunking, embedding, graph extraction)..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data, error = backend_post("/upload", files=files)

        if error:
            st.error(f"Ingestion failed: {error}")
        else:
            st.success(
                f"Ingested '{data['filename']}' in {data['seconds']}s -- "
                f"{data['chunks']} chunks, {data['entities']} new entities, "
                f"{data['relationships']} new relationships."
            )

    st.divider()
    st.header("📊 Knowledge Base")
    stats, error = backend_get("/status")
    if error:
        st.warning(error)
    else:
        c1, c2 = st.columns(2)
        c1.metric("Documents", stats["documents"])
        c2.metric("Chunks", stats["chunks"])
        c3, c4 = st.columns(2)
        c3.metric("Entities", stats["entities"])
        c4.metric("Relationships", stats["relationships"])

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Main: chat interface
# --------------------------------------------------------------------------

st.title("🧠 Hybrid RAG (Vector + Graph)")
st.caption("Upload a PDF in the sidebar, then ask questions grounded in its content.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**[{src['label']}]** *{src['source']}* "
                        f"({src['retrieval']} retrieval)\n\n> {src['snippet']}..."
                    )

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            data, error = backend_post("/query", json={"question": question, "top_k": 5})

        if error:
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error}"})
        else:
            if data.get("guardrail_notice"):
                st.warning(data["guardrail_notice"])
            st.markdown(data["answer"])
            if data.get("sources"):
                with st.expander(f"Sources ({len(data['sources'])})"):
                    for src in data["sources"]:
                        st.markdown(
                            f"**[{src['label']}]** *{src['source']}* "
                            f"({src['retrieval']} retrieval)\n\n> {src['snippet']}..."
                        )
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "sources": data.get("sources", [])}
            )
