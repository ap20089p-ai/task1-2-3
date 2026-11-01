import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_or_update_index, get_vector_store, answer_question, list_supported_documents

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚", layout="wide")
st.title("📚 Document Q&A Assistant")
st.caption("Ask questions about your PDF and TXT documents using Retrieval-Augmented Generation (RAG).")

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")
    st.stop()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Index uploaded documents", type="primary"):
        if not uploads:
            st.warning("Choose at least one PDF or TXT file first.")
        else:
            try:
                document_dir = BASE_DIR / "documents"
                document_dir.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                for upload in uploads:
                    safe_name = Path(upload.name).name
                    destination = document_dir / safe_name
                    destination.write_bytes(upload.getvalue())
                    saved_paths.append(destination)
                with st.spinner("Reading documents and creating embeddings..."):
                    chunks_added = build_or_update_index(saved_paths)
                st.success(f"Indexed {len(saved_paths)} file(s), processing {chunks_added} chunks.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.subheader("Files in documents/")
    current_files = list_supported_documents(BASE_DIR / "documents")
    if current_files:
        for filename in current_files:
            st.write(f"• {filename}")
    else:
        st.caption("No PDF/TXT source files found yet.")
    st.caption("Uploaded files are stored locally. Only upload documents you are allowed to process.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved source candidates"):
                for source in message["sources"]:
                    st.write(f"{source['source']} — page {source['page']}")

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                store = get_vector_store()
                result = answer_question(store, question)
            answer, sources = result["answer"], result["sources"]
            st.markdown(answer)
            if sources:
                with st.expander("Retrieved source candidates"):
                    for source in sources:
                        st.write(f"{source['source']} — page {source['page']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Unable to answer. Check your API key, model access, and index. Details: {exc}")
