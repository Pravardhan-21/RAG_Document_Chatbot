import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from embeddings import get_embedding_model
from utils import VectorStoreError, get_logger, get_vector_store_dir

logger = get_logger("vector_store")

def build_vector_store(chunks: List[Document]) -> FAISS:
    if not chunks:
        raise VectorStoreError("Cannot build a vector store from zero chunks.")
    try:
        embedding_model = get_embedding_model()
        store = FAISS.from_documents(chunks, embedding_model)
        logger.info("Built FAISS index with %d chunk(s).", len(chunks))
        return store
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to build the vector store: {exc}") from exc

def add_documents(store: FAISS, chunks: List[Document]) -> FAISS:
    if not chunks:
        return store
    try:
        store.add_documents(chunks)
        logger.info("Added %d chunk(s) to the existing FAISS index.", len(chunks))
        return store
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to add documents to the vector store: {exc}") from exc

def save_vector_store(store: FAISS, directory: Optional[str] = None) -> str:
    directory = directory or get_vector_store_dir()
    try:
        store.save_local(directory)
        logger.info("Saved FAISS index to '%s'.", directory)
        return directory
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to save the vector store to '{directory}': {exc}") from exc

def load_vector_store(directory: Optional[str] = None) -> Optional[FAISS]:
    directory = directory or get_vector_store_dir()
    if not os.path.isdir(directory):
        return None
    try:
        embedding_model = get_embedding_model()
        store = FAISS.load_local(
            directory, embedding_model, allow_dangerous_deserialization=True
        )
        logger.info("Loaded FAISS index from '%s'.", directory)
        return store
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load vector store from '%s': %s", directory, exc)
        return None

def clear_vector_store(directory: Optional[str] = None) -> None:
    directory = directory or get_vector_store_dir()
    if os.path.isdir(directory):
        shutil.rmtree(directory)
        logger.info("Cleared vector store directory '%s'.", directory)

def get_retriever(store: FAISS, k: int = 4):
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})