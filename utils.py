import os
import logging
from functools import wraps
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
logger = get_logger("utils")

class ConfigError(Exception):
    """Raised when required configuration (e.g. API keys) is missing/invalid."""

class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""

class VectorStoreError(Exception):
    """Raised when the vector store cannot be built, saved, or loaded."""

class ChatbotError(Exception):
    """Raised when the RAG chatbot fails to answer a query."""

def load_environment() -> None:
    load_dotenv(override=False)

def get_groq_api_key() -> str:
    load_environment()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "GROQ_API_KEY is missing. Set it in a .env file (see .env.example) "
            "or as an environment variable before running the app."
        )
    return api_key

def get_groq_model_name() -> str:
    load_environment()
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def get_embedding_model_name() -> str:
    load_environment()
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store_dir() -> str:
    load_environment()
    return os.getenv("VECTOR_STORE_DIR", "faiss_index")

def safe_call(default=None, exceptions=(Exception,), log_prefix="Error"):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as exc:  # noqa: BLE001
                logger.error("%s in %s: %s", log_prefix, func.__name__, exc)
                return default
        return wrapper
    return decorator

def format_sources(source_documents) -> str:
    if not source_documents:
        return ""
    seen = set()
    lines = []
    for doc in source_documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            lines.append(f"- {source} (page {page + 1})")
        else:
            lines.append(f"- {source}")
    return "\n".join(lines)

def bytes_to_human(num_bytes: int) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < step:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} TB"