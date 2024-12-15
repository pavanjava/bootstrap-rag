import os
import uuid
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

context = {
    "user_id": "pavan_mantha",
    "agent_id": "react_agent",
    "run_id": str(uuid.uuid4()),
}

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": os.environ.get('COLLECTION_NAME'),
            "url": os.environ.get('QDRANT_URL'),
            "api_key": os.environ.get('QDRANT_API_KEY'),
            "embedding_model_dims": 768,  # Change this according to your local model's dimensions
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:latest",
            "temperature": 0.2,
            "max_tokens": 1500,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {"model": "nomic-embed-text:latest"},
    },
    "version": "v1.1",
}