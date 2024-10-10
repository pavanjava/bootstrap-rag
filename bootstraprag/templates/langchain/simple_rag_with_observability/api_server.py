from abc import ABC
from dotenv import load_dotenv, find_dotenv
from simple_rag import SimpleRAG
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class SimpleRAGServingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.simpleRAG: SimpleRAG = None
        self.file_path: str = "data/mlops.pdf"
        self.collection_name: str = os.environ.get("COLLECTION_NAME", 'test_collection')
        self.qdrant_url: str = os.environ.get("QDRANT_DB_URL", 'http://localhost:6333')
        self.qdrant_api_key: str = os.environ.get("QDRANT_DB_KEY", 'your_api_key_here')

    def setup(self, devices):
        self.simpleRAG = SimpleRAG(file_path=self.file_path, collection_name=self.collection_name,
                                   qdrant_url=self.qdrant_url, qdrant_api_key=self.qdrant_api_key)

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, query: str):
        return self.simpleRAG.query(user_query=query)

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = SimpleRAGServingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
