import uuid
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SearchParams, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama


class SemanticCache:
    def __init__(self, threshold=0.35):
        # load the data from env
        load_dotenv(find_dotenv())

        self.encoder = SentenceTransformer(model_name_or_path=os.environ.get('model_name_or_path'))
        self.cache_client = QdrantClient(url=os.environ.get('QDRANT_URL'), api_key=os.environ.get('QDRANT_API_KEY'))
        self.cache_collection_name = "cache"
        self.threshold = threshold

        # Create the cache collection
        if not self.cache_client.collection_exists(collection_name=self.cache_collection_name):
            self.cache_client.create_collection(
                collection_name=self.cache_collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

    def get_embedding(self, text):
        return self.encoder.encode([text])[0]

    def search_cache(self, query):
        query_vector = self.get_embedding(query)
        search_result = self.cache_client.search(
            collection_name=self.cache_collection_name,
            query_vector=query_vector,
            limit=1,
            search_params=SearchParams(hnsw_ef=128)
        )
        if search_result and search_result[0].score > self.threshold:
            return search_result[0].payload['response']
        return None

    def add_to_cache(self, query, response):
        query_vector = self.get_embedding(query)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={"query": query, "response": response}
        )
        self.cache_client.upsert(
            collection_name=self.cache_collection_name,
            points=[point]
        )

    def get_response(self, query, compute_response_func):
        cached_response = self.search_cache(query)
        if cached_response:
            return cached_response
        _response = compute_response_func(query)
        self.add_to_cache(query, _response)
        return _response


# Example usage
def compute_response(query: str):
    llm = Ollama(model=os.environ.get('OLLAMA_MODEL'), base_url=os.environ.get('OLLAMA_BASE_URL'))
    # Create a user message
    user_message = ChatMessage(
        role=MessageRole.USER,
        content=query
    )

    # Generate a response from the assistant
    assistant_message = llm.chat(messages=[user_message])
    return f"Computed response for: {query} is {assistant_message}"


# semantic_cache = SemanticCache(threshold=0.8)
# query = "What is the capital of France?"
# response = semantic_cache.get_response(query, compute_response)
# print(response)
