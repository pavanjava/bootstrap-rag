import os
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from dotenv import load_dotenv, find_dotenv
import qdrant_client


class CitationQueryEngineRAG:
    def __init__(self, data_path="data", required_exts=None):
        """
        Initializes the MLOpsQueryEngine with data path and environment variables.
        """
        if required_exts is None:
            required_exts = ['.pdf', '.txt']

        load_dotenv(find_dotenv())

        self.data_path = data_path
        self.required_exts = required_exts

        # Initialize settings
        self._initialize_settings()

        # Load documents
        self.documents = self._load_documents()

        # Initialize vector store
        self.vector_store = self._initialize_vector_store()

        # Create storage context
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents=self.documents, storage_context=self.storage_context
        )

        # Initialize query engine
        self.query_engine = CitationQueryEngine.from_args(
            self.index,
            similarity_top_k=3,
            citation_chunk_size=256,
        )

    def _initialize_settings(self):
        """
        Initialize LLM and embedding model settings.
        """
        Settings.llm = Ollama(
            model=os.environ.get("OLLAMA_LLM_MODEL"),
            base_url=os.environ.get("OLLAMA_BASE_URL")
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=os.environ.get("OLLAMA_EMBED_MODEL"),
            base_url=os.environ.get("OLLAMA_BASE_URL")
        )

    def _load_documents(self):
        """
        Loads documents from the specified directory.
        """
        return SimpleDirectoryReader(
            self.data_path, required_exts=self.required_exts
        ).load_data(show_progress=True)

    def _initialize_vector_store(self):
        """
        Initializes the Qdrant vector store.
        """
        client = qdrant_client.QdrantClient(
            url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY']
        )
        return QdrantVectorStore(
            client=client,
            collection_name=os.environ['COLLECTION_NAME']
        )

    def query(self, question):
        """
        Queries the MLOps query engine.
        """
        return self.query_engine.query(question)
