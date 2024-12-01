from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langfuse.llama_index import LlamaIndexInstrumentor
from dotenv import load_dotenv, find_dotenv
import qdrant_client
import os

# Load environment variables
load_dotenv(find_dotenv())

# instrumenting observability
instrumentor = LlamaIndexInstrumentor()
instrumentor.start()


class SubQuestionQueryEngineAgent:
    def __init__(self):
        # Initialize debug handler and settings
        self._setup_settings()

        # Initialize Qdrant clients
        self.client = qdrant_client.QdrantClient(
            url=os.environ['DB_URL'],
            api_key=os.environ['DB_API_KEY']
        )
        self.aclient = qdrant_client.AsyncQdrantClient(
            url=os.environ['DB_URL'],
            api_key=os.environ['DB_API_KEY']
        )

        # Initialize query engine
        self.query_engine = None

    def _setup_settings(self):
        """Configure LlamaIndex settings"""
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])

        Settings.callback_manager = callback_manager
        Settings.llm = OpenAI(
            model=os.environ.get("OPENAI_EMBED_MODEL"),
            temperature=0.0
        )
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size = int(os.environ.get("CHUNK_SIZE"))
        Settings.chunk_overlap = int(os.environ.get("CHUNK_OVERLAP"))

    def load_and_index_documents(self, input_dir="data"):
        """Load documents and create vector store index"""
        # Load documents
        orthodontics_docs = SimpleDirectoryReader(input_dir=input_dir).load_data(
            show_progress=True
        )

        # Setup vector store and index
        vector_store = QdrantVectorStore(
            client=self.client,
            aclient=self.aclient,
            collection_name=os.environ.get("COLLECTION_NAME")
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        if not self.client.collection_exists(collection_name=os.environ.get("COLLECTION_NAME")):
            vector_query_engine = VectorStoreIndex.from_documents(
                documents=orthodontics_docs,
                storage_context=storage_context
            ).as_query_engine()
        else:
            vector_query_engine = VectorStoreIndex.from_vector_store(
                vector_store=vector_store).as_query_engine()

        # Setup query engine tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="orthodontics_tool",
                    description="guide on the orthodontics",
                ),
            ),
        ]

        # Initialize SubQuestionQueryEngine
        self.query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools
        )

    def query(self, question: str):
        """Execute a query and return the response"""
        if self.query_engine is None:
            raise ValueError("Query engine not initialized. Call load_and_index_documents first.")

        return self.query_engine.query(question)
