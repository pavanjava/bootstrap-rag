from abc import ABC
from dotenv import load_dotenv, find_dotenv
from openai import base_url

from core_advanced_rag import RetrievalAugmentationGenerationUsingHyDE
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class SimpleRAGServingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.advanced_rag: RetrievalAugmentationGenerationUsingHyDE = None
        self.FILE_PATH = 'data/mlops.pdf'
        self.COLLECTION_NAME = os.environ.get('COLLECTION_NAME')
        self.VECTOR_NAME = os.environ.get('VECTOR_NAME')
        self.PROMPT_TEMPLATE = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        self.LLM_MODEL = os.environ.get('LLM_MODEL')
        self.EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
        self.QDRANT_URL = os.environ.get('QDRANT_URL')
        self.QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
        self.OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL')

    def setup(self, devices):
        self.advanced_rag = RetrievalAugmentationGenerationUsingHyDE(
            file_path=self.FILE_PATH,
            collection_name=self.COLLECTION_NAME,
            vector_name=self.VECTOR_NAME,
            prompt_template=self.PROMPT_TEMPLATE,
            llm_model=self.LLM_MODEL,
            embedding_model=self.EMBEDDING_MODEL,
            qdrant_url=self.QDRANT_URL,
            qdrant_api_key=self.QDRANT_API_KEY,
            base_url=self.OLLAMA_BASE_URL
        )
        # Load Documents
        documents = self.advanced_rag.load_documents()

        # Get Embeddings
        embeddings = self.advanced_rag.get_embeddings()

        # Setup Qdrant and Add Documents
        self.advanced_rag.setup_qdrant_collection(embeddings=embeddings, documents=documents)

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, query: str):
        return self.advanced_rag.execute_pipeline(user_query=query)

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = SimpleRAGServingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
