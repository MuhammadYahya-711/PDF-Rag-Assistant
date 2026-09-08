import os
from io import BytesIO

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.write(
    "Upload a PDF and ask questions about its content using "
    "Retrieval-Augmented Generation (RAG)."
)


# -----------------------------
# SETTINGS
# -----------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# GET GROQ API KEY
# -----------------------------

def get_groq_api_key():

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:

            text = " ".join(text.split())

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# -----------------------------
# CREATE CHUNKS
# -----------------------------

def create_chunks(pages):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"]
                    }
                )

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# -----------------------------
# CREATE CHROMA DATABASE
# -----------------------------

def create_vector_database(chunks, embedding_model):

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="pdf_documents"
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    ).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return client, collection


# -----------------------------
# SEARCH VECTOR DATABASE
# -----------------------------

def search_documents(
    question,
    collection,
    embedding_model,
    top_k=TOP_K
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    retrieved_chunks = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved_chunks.append(
            {
                "text": document,
                "page": metadata["page"]
            }
        )

    return retrieved_chunks


# -----------------------------
# CREATE CONTEXT
# -----------------------------

def create_context(retrieved_chunks):

    context = ""

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context += (
            f"\n\n--- Source {i} "
            f"(PDF Page {chunk['page']}) ---\n"
        )

        context += chunk["text"]

    return context


# -----------------------------
# ASK GROQ
# -----------------------------

def ask_groq(question, context, api_key):

    client = Groq(
        api_key=api_key
    )

    system_prompt = """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the PDF context.

Rules:

1. Do not invent information.
2. If the answer is not present in the PDF, say:
   "I could not find the answer in the uploaded PDF."
3. Give clear and concise answers.
4. Mention PDF page numbers when possible.
"""

    user_prompt = f"""
PDF CONTEXT:

{context}


USER QUESTION:

{question}
"""

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=2,
        max_value=10,
        value=5
    )

    st.markdown("### 🔧 Technology")

    st.write(
        "📄 PDF: PyPDF"
    )

    st.write(
        "🧠 Embeddings: Sentence Transformers"
    )

    st.write(
        "🗄️ Vector DB: ChromaDB"
    )

    st.write(
        "🤖 LLM: Groq"
    )

    st.write(
        f"Model: `{GROQ_MODEL}`"
    )


# -----------------------------
# API KEY CHECK
# -----------------------------

groq_api_key = get_groq_api_key()

if not groq_api_key:

    st.warning(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Add your Groq API key in Streamlit Cloud → "
        "Settings → Secrets."
    )

    st.code(
        'GROQ_API_KEY = "your_api_key_here"',
        language="toml"
    )

    st.stop()


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# -----------------------------
# PROCESS PDF
# -----------------------------

if uploaded_file:

    st.write(
        f"**Selected file:** {uploaded_file.name}"
    )

    if st.button(
        "🚀 Process PDF",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Processing your PDF..."
        ):

            try:

                # Extract text
                pages = extract_text_from_pdf(
                    uploaded_file
                )

                if not pages:

                    st.error(
                        "No readable text was found "
                        "in this PDF."
                    )

                    st.stop()

                # Create chunks
                chunks = create_chunks(
                    pages
                )

                # Load embedding model
                embedding_model = (
                    load_embedding_model()
                )

                # Create vector DB
                client, collection = (
                    create_vector_database(
                        chunks,
                        embedding_model
                    )
                )

                # Save in session
                st.session_state[
                    "collection"
                ] = collection

                st.session_state[
                    "embedding_model"
                ] = embedding_model

                st.session_state[
                    "document_name"
                ] = uploaded_file.name

                st.session_state[
                    "chunks"
                ] = chunks

                st.session_state[
                    "pages"
                ] = pages

                st.success(
                    "✅ PDF processed successfully!"
                )

                st.write(
                    f"Pages extracted: **{len(pages)}**"
                )

                st.write(
                    f"Chunks created: **{len(chunks)}**"
                )

            except Exception as error:

                st.error(
                    f"Error while processing PDF: {error}"
                )


# -----------------------------
# QUESTION ANSWERING
# -----------------------------

if "collection" in st.session_state:

    st.divider()

    st.subheader(
        "💬 Ask a question about your PDF"
    )

    question = st.chat_input(
        "Ask something about your document..."
    )

    if question:

        # Show question
        with st.chat_message("user"):

            st.write(question)

        # Search database
        with st.spinner(
            "Searching your document..."
        ):

            retrieved_chunks = search_documents(
                question,
                st.session_state["collection"],
                st.session_state["embedding_model"],
                top_k
            )

        # Create context
        context = create_context(
            retrieved_chunks
        )

        # Ask Groq
        with st.chat_message("assistant"):

            with st.spinner(
                "Generating answer..."
            ):

                try:

                    answer = ask_groq(
                        question,
                        context,
                        groq_api_key
                    )

                    st.markdown(answer)

                except Exception as error:

                    st.error(
                        f"Groq error: {error}"
                    )

        # Sources
        st.subheader(
            "📚 Retrieved Sources"
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            with st.expander(
                f"Source {i} — PDF Page {chunk['page']}"
            ):

                st.write(
                    chunk["text"]
                )

else:

    st.info(
        "👆 Upload a PDF and click "
        "**Process PDF** to start."
    )
