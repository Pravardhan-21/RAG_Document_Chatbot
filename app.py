import streamlit as st
from pdf_loader import load_pdfs
from text_splitter import split_documents
from vector_store import (
    add_documents,
    build_vector_store,
    clear_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
)
from rag_chatbot import RAGChatbot
from utils import (
    ChatbotError,
    ConfigError,
    PDFProcessingError,
    VectorStoreError,
    format_sources,
    get_logger,
)

logger = get_logger("app")

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="wide")

def init_state():
    defaults = {
        "vector_store": None,
        "chatbot": None,
        "messages": [],  # list of {"role": ..., "content": ..., "sources": ...}
        "processed_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()
if st.session_state.vector_store is None:
    restored = load_vector_store()
    if restored is not None:
        st.session_state.vector_store = restored

with st.sidebar:
    st.title("📄 RAG Document Chatbot")
    st.caption("Upload PDFs, then ask questions about their content.")
    st.subheader("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        process_clicked = st.button("Process documents", type="primary", use_container_width=True)
    with col_b:
        clear_clicked = st.button("Clear database", use_container_width=True)
    if clear_clicked:
        clear_vector_store()
        st.session_state.vector_store = None
        st.session_state.chatbot = None
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.success("Vector database cleared. Upload new PDFs to rebuild it.")

    if process_clicked:
        if not uploaded_files:
            st.warning("Please choose at least one PDF file first.")
        else:
            with st.spinner("Extracting text, chunking, and embedding..."):
                try:
                    docs = load_pdfs(uploaded_files, [f.name for f in uploaded_files])
                    chunks = split_documents(docs)
                    if st.session_state.vector_store is None:
                        store = build_vector_store(chunks)
                    else:
                        store = add_documents(st.session_state.vector_store, chunks)
                    save_vector_store(store)
                    st.session_state.vector_store = store
                    st.session_state.chatbot = None  # force re-init with new retriever
                    st.session_state.processed_files.extend([f.name for f in uploaded_files])
                    st.success(f"Processed {len(uploaded_files)} file(s) into {len(chunks)} chunks.")
                except PDFProcessingError as exc:
                    st.error(f"PDF error: {exc}")
                except VectorStoreError as exc:
                    st.error(f"Vector store error: {exc}")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Unexpected error while processing documents")
                    st.error(f"Unexpected error: {exc}")
    if st.session_state.processed_files:
        st.subheader("Indexed files")
        for name in sorted(set(st.session_state.processed_files)):
            st.write(f"- {name}")
    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.reset_memory()
    with st.expander("⚙️ Settings"):
        top_k = st.slider("Chunks retrieved per question (k)", 2, 10, 4)
        show_sources = st.checkbox("Show source document & page", value=True)

st.header("Chat with your documents")
if st.session_state.vector_store is None:
    st.info("👈 Upload one or more PDFs and click **Process documents** to get started.")
else:
    if st.session_state.chatbot is None:
        try:
            retriever = get_retriever(st.session_state.vector_store, k=top_k)
            st.session_state.chatbot = RAGChatbot(retriever=retriever, k=top_k)
        except ConfigError as exc:
            st.error(
                f"Configuration error: {exc}\n\n"
                "Set GROQ_API_KEY in a `.env` file (see `.env.example`) and restart the app."
            )
            st.stop()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    st.text(msg["sources"])
    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chatbot.ask(question)
                    answer = result["answer"]
                    sources = format_sources(result["source_documents"]) if show_sources else ""
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            st.text(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except ChatbotError as exc:
                    error_msg = f"⚠️ {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Unexpected error while answering question")
                    error_msg = f"⚠️ Unexpected error: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})