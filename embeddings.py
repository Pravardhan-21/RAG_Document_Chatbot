from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from utils import get_embedding_model_name, get_logger
logger = get_logger("embeddings")

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    model_name = get_embedding_model_name()
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )