import os
import re
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.write(
    "Upload a PDF, index its content with open-source embeddings + FAISS, "
    "and ask questions using an open-weight model through Groq."
)

# -----------------------------
# Constants
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def create_groq_client(api_key):
    return Groq(api_key=api_key)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(uploaded_file):
    """Extract text page-by-page from an uploaded PDF."""
    pdf_bytes = uploaded_file.getvalue()
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# -----------------------------
# Chunking
# -----------------------------
def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping character-based chunks while keeping page metadata."""
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "page": page["page"],
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            start = max(end - overlap, start + 1)

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_faiss_index(chunks, embedding_model):
    """Create normalized embeddings and store them in a FAISS cosine-similarity index."""
    texts = [chunk["text"] for chunk in chunks]

    # SentenceTransformer handles tokenization and creates vector embeddings.
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # Normalize vectors so inner product ~= cosine similarity.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_index(query, index, chunks, embedding_model, top_k=TOP_K):
    """Retrieve the most relevant chunks for a query."""
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):
        if index_number == -1:
            continue

        result = chunks[index_number].copy()
        result["score"] = float(score)
        results.append(result)

    return results


# -----------------------------
# RAG prompt
# -----------------------------
def build_context(results):
    """Format retrieved chunks into a source context for the LLM."""
    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {result['page']}]\n{result['text']}"
        )

    return "\n\n".join(context_parts)


def ask_groq(client, question, context):
    """Ask Groq to answer using only the retrieved PDF context."""
    system_prompt = """You are a PDF question-answering assistant using Retrieval-Augmented Generation (RAG).

Rules:
1. Answer using the supplied PDF context.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain the answer, clearly say that the answer was not found in the uploaded PDF.
4. Keep the answer clear and useful.
5. Mention the relevant PDF page number(s) when possible.
"""

    user_prompt = f"""PDF CONTEXT:
{context}

QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=TOP_K,
        help="How many relevant PDF chunks are sent to the model.",
    )

    st.info(
        "Embedding model: all-MiniLM-L6-v2\n\n"
        "Vector database: FAISS\n\n"
        f"Groq model: {GROQ_MODEL}"
    )

    st.caption(
        "The PDF and FAISS index are kept in app memory for the current session. "
        "They are rebuilt when the app needs to process a new PDF."
    )


# -----------------------------
# API key
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not groq_api_key:
    st.warning("Add your Groq API key before using the app.")
    st.code("GROQ_API_KEY=your_api_key_here", language="text")
    st.stop()

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Text-based PDFs work best. Scanned/image-only PDFs may need OCR, which this version does not include.",
)

if uploaded_file:
    st.caption(
        f"Selected: {uploaded_file.name} • "
        f"{uploaded_file.size / 1024:.1f} KB"
    )

    if st.button("🔎 Process PDF", type="primary", use_container_width=True):
        with st.spinner("Extracting, chunking, embedding, and indexing..."):
            try:
                pages = extract_pdf_text(uploaded_file)

                if not pages:
                    st.error(
                        "No selectable text was found in this PDF. "
                        "This version does not include OCR for scanned PDFs."
                    )
                    st.stop()

                chunks = create_chunks(pages)
                embedding_model = load_embedding_model()
                index = build_faiss_index(chunks, embedding_model)

                st.session_state["document_name"] = uploaded_file.name
                st.session_state["pages"] = pages
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index

                st.session_state["chat_history"] = []

                st.success(
                    f"PDF processed successfully: {len(pages)} pages, "
                    f"{len(chunks)} chunks indexed."
                )

            except Exception as error:
                st.error(f"Could not process the PDF: {error}")


# -----------------------------
# Show document status
# -----------------------------
if "index" in st.session_state:
    st.success(
        f"✅ Ready: {st.session_state['document_name']} "
        f"({len(st.session_state['pages'])} pages, "
        f"{len(st.session_state['chunks'])} chunks)"
    )

    st.divider()
    st.subheader("💬 Ask a question about your PDF")

    # Display previous chat messages.
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the uploaded PDF...")

    if question:
        st.session_state["chat_history"].append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and generating an answer..."):
                try:
                    embedding_model = load_embedding_model()

                    results = search_index(
                        question,
                        st.session_state["index"],
                        st.session_state["chunks"],
                        embedding_model,
                        top_k=top_k,
                    )

                    context = build_context(results)

                    client = create_groq_client(groq_api_key)
                    answer = ask_groq(client, question, context)

                    st.markdown(answer)

                    st.caption(
                        "Retrieved sources: "
                        + ", ".join(
                            f"page {result['page']} ({result['score']:.2f})"
                            for result in results
                        )
                    )

                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as error:
                    st.error(f"Something went wrong: {error}")

    with st.expander("🔍 View retrieved chunks"):
        if question:
            embedding_model = load_embedding_model()
            results = search_index(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
                embedding_model,
                top_k=top_k,
            )

            for i, result in enumerate(results, start=1):
                st.markdown(
                    f"**Source {i} — Page {result['page']} — "
                    f"Similarity: {result['score']:.3f}**"
                )
                st.write(result["text"])
                st.divider()
else:
    st.info("Upload a PDF and click **Process PDF** to build the RAG index.")
