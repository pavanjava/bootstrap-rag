import os
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading the configuration")

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": os.environ.get("COLLECTION_NAME"),
            "url": os.environ.get("QDRANT_URL"),
            "api_key": os.environ.get("QDRANT_API_KEY"),
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": os.environ.get("OLLAMA_MODEL"),
            "temperature": 0,
            "max_tokens": 8000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": os.environ.get("OLLAMA_EMBED_MODEL"),
            "ollama_base_url": "http://localhost:11434",
        },
    },
}