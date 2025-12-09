import os
import sys
from pathlib import Path

import streamlit as st

# Allow imports from ./src when running `streamlit run app.py`
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from build_index import build_faiss_index
from chunker import chunk_text
from gemini_client import embed_text
from pdf_loader import load_pdf_text
from rag_answer import rag_answer

PDF_PATH = BASE_DIR / "data" / "jio_info.pdf"


def require_api_key():
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "Missing Google API key. Set the GOOGLE_API_KEY environment variable "
            "before starting Streamlit."
        )
        st.stop()


@st.cache_resource(show_spinner="Building vector index…")
def prepare_index():
    text = load_pdf_text(str(PDF_PATH))
    chunks = chunk_text(text)

    embeddings = []
    for chunk in chunks:
        embeddings.append(embed_text(chunk))

    index, _ = build_faiss_index(embeddings)
    metadata = [{"text": c, "source": str(PDF_PATH)} for c in chunks]
    return index, metadata


def show_sidebar(metadata):
    st.sidebar.header("Knowledge Base")
    st.sidebar.write(f"Source file: `{PDF_PATH.name}`")
    st.sidebar.write(f"Indexed chunks: {len(metadata)}")
    st.sidebar.markdown(
        "Tip: Ask concise questions about the PDF, e.g. "
        "\"What are the main features offered?\""
    )


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("RAG Chatbot")
    st.caption("Powered by Gemini embeddings + retrieval over the provided PDF.")

    require_api_key()

    try:
        index, metadata = prepare_index()
    except Exception as exc:
        st.error(f"Failed to prepare the index: {exc}")
        st.stop()

    show_sidebar(metadata)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the PDF")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = rag_answer(user_input, index, metadata)
                except Exception as exc:  # surface API/other failures nicely
                    answer = f"Sorry, I ran into an error: {exc}"
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

