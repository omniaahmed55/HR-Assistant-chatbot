import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# API
# =========================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# Models
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# =========================
# Memory
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_buffer" not in st.session_state:
    st.session_state.conversation_buffer = []

MAX_MEMORY = 6

# =========================
# LLM
# =========================

def call_llm(prompt):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# =========================
# UI
# =========================

st.title("📄 PDF Chat Assistant")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================
# Process PDF
# =========================

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text

    # Chunking
    def chunk_text(text, chunk_size=500, overlap=50):

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunks.append(text[start:end])

            start += chunk_size - overlap

        return chunks

    chunks = chunk_text(full_text)

    # Chroma
    client_db = chromadb.EphemeralClient()

    collection = client_db.get_or_create_collection(
        name="my_data"
    )

    embds = embed_model.encode(
        chunks
    ).tolist()

    ids = [
        str(i)
        for i in range(len(chunks))
    ]

    try:
        collection.add(
            documents=chunks,
            embeddings=embds,
            ids=ids
        )
    except:
        pass

    # Show chat history
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User question
    query = st.chat_input(
        "Ask about the PDF..."
    )

    if query:

        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # Retrieval
        embd_q = embed_model.encode(
            query
        ).tolist()

        results = collection.query(
            query_embeddings=[embd_q],
            n_results=3
        )

        retrieved = "\n---\n".join(
            results["documents"][0]
        )

        history = "\n".join(
            st.session_state.conversation_buffer
        )

        prompt = f"""
You are a helpful AI assistant.

Conversation History:
{history}

PDF Context:
{retrieved}

Instructions:
1. Answer from PDF context first.
2. Use history if needed.
3. If not found in PDF, answer normally.
4. Keep answers concise.

Question:
{query}

Answer:
"""

        answer = call_llm(prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.session_state.conversation_buffer.append(
            f"Human: {query}"
        )

        st.session_state.conversation_buffer.append(
            f"AI: {answer}"
        )

        st.session_state.conversation_buffer = (
            st.session_state.conversation_buffer[-MAX_MEMORY:]
        )

else:

    st.info(
        "Upload a PDF from the sidebar to start chatting."
    )