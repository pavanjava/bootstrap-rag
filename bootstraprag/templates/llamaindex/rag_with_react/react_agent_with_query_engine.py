from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.base.response.schema import Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
from dotenv import load_dotenv, find_dotenv
from typing import Union
import qdrant_client
import logging
import os

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class ReActWithQueryEngine:

    RESPONSE_TYPE = Union[
        Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
    ]

    def __init__(self, input_dir: str, similarity_top_k: int = 3):
        self.index_loaded = False
        self.similarity_top_k = similarity_top_k
        self.input_dir = input_dir
        self._index = None
        self._engine = None
        self.agent: ReActAgent = None
        self.query_engine_tools = []

        # use your prefered vector embeddings model
        logger.info("initializing the OllamaEmbedding")
        embed_model = OllamaEmbedding(model_name=os.environ['OLLMA_EMBED_MODEL'],
                                      base_url=os.environ['OLLAMA_BASE_URL'])
        # openai embeddings, embedding_model_name="text-embedding-3-large"
        # embed_model = OpenAIEmbedding(embed_batch_size=10, model=embedding_model_name)

        # use your prefered llm
        llm = Ollama(model=os.environ['OLLMA_LLM_MODEL'], base_url=os.environ['OLLAMA_BASE_URL'], request_timeout=600)
        # llm = OpenAI(model="gpt-4o")

        logger.info("initializing the global settings")
        Settings.embed_model = embed_model
        Settings.llm = llm

        # Create a local Qdrant vector store
        logger.info("initializing the vector store related objects")
        client = qdrant_client.QdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])
        self.vector_store = QdrantVectorStore(client=client, collection_name=os.environ['COLLECTION_NAME'])
        self._load_data_and_create_engine()

    def _load_data_and_create_engine(self):
        try:
            self._index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
            self.index_loaded = True
        except Exception as e:
            self.index_loaded = False

        if not self.index_loaded:
            # load data
            _docs = SimpleDirectoryReader(input_dir=self.input_dir).load_data()

            # build and persist index
            self._index = VectorStoreIndex.from_documents(documents=_docs)

        self._engine = self._index.as_query_engine(similarity_top_k=self.similarity_top_k)
        self._create_query_engine_tools()

    def _create_query_engine_tools(self):
        # can have more than one as per the requirement
        self.query_engine_tools.append(
            QueryEngineTool(
                query_engine=self._engine,
                metadata=ToolMetadata(
                    name="test_tool_engine",  # change this accordingly
                    description=(
                        "Provides information about user query based on the information that you have. "
                        "Use a detailed plain text question as input to the tool."
                    ),
                ),
            )
        )
        self._create_react_agent()

    def _create_react_agent(self):
        # [Optional] Add Context
        # context = """\
        # You are a stock market sorcerer who is an expert on the companies Lyft and Uber.\
        #     You will answer questions about Uber and Lyft as in the persona of a sorcerer \
        #     and veteran stock market investor.
        # """
        self.agent = ReActAgent.from_tools(
            self.query_engine_tools,
            llm=Settings.llm,
            verbose=True,
            # context=context
        )

    def query(self, user_query: str) -> RESPONSE_TYPE:
        return self.agent.query(str_or_query_bundle=user_query)
