import os
from typing import Dict, List

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv, find_dotenv
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore


class RecursiveAgentManager:
    def __init__(self, agent_names: List[str], data_dir: str = 'data'):
        """
        Initialize the AgentManager with a list of agent names and optional data directory path.

        Args:
            agent_names (List[str]): List of agent names to initialize
            data_dir (str): Directory containing the PDF files (default: 'data')
        """
        self.agent_names = agent_names
        self.data_dir = data_dir
        self.document_data = {}
        self.agents = {}
        self.query_engine = None
        self.client: qdrant_client.QdrantClient = None

        # Load environment variables
        load_dotenv(find_dotenv())

        # Initialize settings
        self._initialize_settings()

        # Setup components
        self._load_documents()
        self._setup_vector_store()
        self._build_agents()
        self._setup_query_engine()

    def _initialize_settings(self):
        """Initialize LlamaIndex settings."""
        Settings.llm = OpenAI(model=os.environ.get("OPENAI_EMBED_MODEL"), temperature=0.0)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size = int(os.environ.get("CHUNK_SIZE"))
        Settings.chunk_overlap = int(os.environ.get("CHUNK_OVERLAP"))

    def _load_documents(self):
        """Load PDF documents for each agent."""
        for agent_name in self.agent_names:
            self.document_data[agent_name] = SimpleDirectoryReader(
                input_files=[f'{self.data_dir}/{agent_name}.pdf']
            ).load_data()

    def _setup_vector_store(self):
        """Setup Qdrant vector store and storage context."""
        client = qdrant_client.QdrantClient(
            url=os.environ['DB_URL'],
            api_key=os.environ['DB_API_KEY']
        )
        self.client = client
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=os.environ['COLLECTION_NAME']
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _create_query_engine_tools(self, agent_name: str, vector_index: VectorStoreIndex,
                                   summary_index: SummaryIndex) -> List[QueryEngineTool]:
        """Create query engine tools for an agent."""
        return [
            QueryEngineTool(
                query_engine=vector_index.as_query_engine(),
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=f"Useful for retrieving specific context from {agent_name}"
                ),
            ),
            QueryEngineTool(
                query_engine=summary_index.as_query_engine(),
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=f"Useful for summarization questions related to {agent_name}"
                ),
            ),
        ]

    def _build_agents(self):
        """Build agents with their respective tools."""
        for agent_name in self.agent_names:

            if not self.client.collection_exists(collection_name=os.environ.get("COLLECTION_NAME")):
                # Build indices
                vector_index = VectorStoreIndex.from_documents(
                    self.document_data[agent_name],
                    storage_context=self.storage_context
                )
            else:
                vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

            summary_index = SummaryIndex.from_documents(
                self.document_data[agent_name],
                storage_context=self.storage_context
            )

            # Create tools
            query_engine_tools = self._create_query_engine_tools(
                agent_name, vector_index, summary_index
            )

            # Build agent
            function_llm = OpenAI(model=os.environ.get("OPENAI_EMBED_MODEL"))
            agent = OpenAIAgent.from_tools(
                query_engine_tools,
                llm=function_llm,
                verbose=True,
            )

            self.agents[agent_name] = agent

    def _setup_query_engine(self):
        """Setup the top-level query engine."""
        objects = []
        for agent_name in self.agent_names:
            data_summary = (
                f"This content contains specific data about {agent_name}. Use "
                f"this index if you need to lookup specific facts about {agent_name}.\n"
                f"Do not use this index if you want to analyze different topic "
                f"other than {agent_name}"
            )
            node = IndexNode(
                text=data_summary,
                index_id=agent_name,
                obj=self.agents[agent_name]
            )
            objects.append(node)

        vector_index = VectorStoreIndex(objects=objects)
        self.query_engine = vector_index.as_query_engine(
            similarity_top_k=1,
            verbose=True
        )

    def query(self, question: str) -> str:
        """
        Query the agent system with a question.

        Args:
            question (str): The question to ask

        Returns:
            str: The response from the query engine
        """
        if self.query_engine is None:
            raise RuntimeError("Query engine not initialized")
        return self.query_engine.query(question)
