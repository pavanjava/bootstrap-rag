import openai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings, Response,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.postprocessor import PrevNextNodePostprocessor
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv, find_dotenv
import os
import logging

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


class PrevNextPostprocessorDemo:
    def __init__(self, openai_api_key: str = None, data_directory: str = "data", chunk_size: int = 512):
        """
        Initialize the PrevNextPostprocessorDemo class.

        Args:
            openai_api_key (str): Your OpenAI API key.
            data_directory (str): Path to the directory containing documents.
            chunk_size (int): Chunk size for document parsing.
        """
        if openai_api_key is not None:
            openai.api_key = openai_api_key
        else:
            Settings.llm = Ollama(model=os.environ.get("OLLAMA_LLM_MODEL"), base_url=os.environ.get("OLLAMA_BASE_URL"))
            Settings.embed_model = OllamaEmbedding(model_name=os.environ.get("OLLAMA_EMBED_MODEL"), base_url=os.environ.get("OLLAMA_BASE_URL"))

        self.data_directory = data_directory
        self.chunk_size = chunk_size
        self.documents = None
        self.nodes = None
        self.docstore = None
        self.storage_context = None
        self.index: VectorStoreIndex = None

    def load_documents(self):
        """Load documents from the specified directory."""
        self.documents = SimpleDirectoryReader(self.data_directory).load_data()

    def parse_documents_to_nodes(self):
        """Parse documents into nodes and add them to a document store."""
        # Update settings for chunk size
        Settings.chunk_size = self.chunk_size

        # Parse nodes from documents
        self.nodes = Settings.node_parser.get_nodes_from_documents(self.documents)

        # Create and populate the document store
        self.docstore = SimpleDocumentStore()
        self.docstore.add_documents(self.nodes)

        # Create a storage context
        self.storage_context = StorageContext.from_defaults(docstore=self.docstore)

    def build_index(self):
        """Build a VectorStoreIndex using the parsed nodes."""
        if not self.nodes:
            raise ValueError("Nodes are not parsed. Call parse_documents_to_nodes first.")
        self.index = VectorStoreIndex(self.nodes, storage_context=self.storage_context)

    def query_with_postprocessor(self, query, num_nodes=4, top_k=1, response_mode="tree_summarize"):
        """
        Query the index with a PrevNextNodePostprocessor.

        Args:
            query (str): The query string.
            num_nodes (int): Number of adjacent nodes to include in postprocessing.
            top_k (int): Number of top results to consider.
            response_mode (str): Response mode for the query engine.

        Returns:
            str: Query response.
        """
        if not self.index:
            raise ValueError("Index is not built. Call build_index first.")

        node_postprocessor = PrevNextNodePostprocessor(docstore=self.docstore, num_nodes=num_nodes)

        query_engine: BaseQueryEngine = self.index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=[node_postprocessor],
            response_mode=response_mode,
        )
        response: Response = query_engine.query(query)
        return response

    def query_without_postprocessor(self, query, top_k=1, response_mode="tree_summarize"):
        """
        Query the index without any postprocessor.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to consider.
            response_mode (str): Response mode for the query engine.

        Returns:
            str: Query response.
        """
        if not self.index:
            raise ValueError("Index is not built. Call build_index first.")

        query_engine = self.index.as_query_engine(similarity_top_k=top_k, response_mode=response_mode)
        response = query_engine.query(query)
        return response
