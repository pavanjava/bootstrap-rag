from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
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
from llama_index.memory.mem0 import Mem0Memory
import qdrant_client
import logging
import os
import uuid

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class ReActWithQueryEngine:
    RESPONSE_TYPE = Union[
        Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
    ]

    def __init__(self, input_dir: str, similarity_top_k: int = 3, chunk_size: int = 128, chunk_overlap: int = 100,
                 show_progress: bool = False, no_of_iterations: int = 20, required_exts: list[str] = ['.pdf', '.txt']):
        self.index_loaded = False
        self.similarity_top_k = similarity_top_k
        self.input_dir = input_dir
        self._index = None
        self._engine = None
        self.agent: ReActAgent = None
        self.query_engine_tools = []
        self.show_progress = show_progress
        self.no_of_iterations = no_of_iterations
        self.required_exts = required_exts

        # the mem0 configs for llamaindex
        context = {
            "user_id": "pavan_mantha",
            "agent_id": "react_agent",
            "run_id": str(uuid.uuid4()),
        }

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "react-agent-with-memory",
                    "url": "http://localhost:6333",
                    "api_key": "th3s3cr3tk3y",
                    "embedding_model_dims": 768
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
                "config": {
                    "model": "nomic-embed-text:latest"
                },
            }
        }

        self.memory = Mem0Memory.from_config(
            context=context,
            config=config,
            search_msg_limit=10  # default is 5
        )

        # use your prefered vector embeddings model
        logger.info("initializing the OllamaEmbedding")
        embed_model = OllamaEmbedding(model_name=os.environ['OLLAMA_EMBED_MODEL'],
                                      base_url=os.environ['OLLAMA_BASE_URL'])
        # openai embeddings, embedding_model_name="text-embedding-3-large"
        # embed_model = OpenAIEmbedding(embed_batch_size=10, model=embedding_model_name)

        # use your prefered llm
        llm = Ollama(model=os.environ['OLLAMA_LLM_MODEL'], base_url=os.environ['OLLAMA_BASE_URL'], request_timeout=600)
        # llm = OpenAI(model="gpt-4o")

        logger.info("initializing the global settings")
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        # Create a local Qdrant vector store
        logger.info("initializing the vector store related objects")
        self.client: qdrant_client.QdrantClient = qdrant_client.QdrantClient(url=os.environ['QDRANT_URL'],
                                                                             api_key=os.environ['QDRANT_API_KEY'])
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=os.environ['COLLECTION_NAME'])
        self._load_data_and_create_engine()

    def _load_data_and_create_engine(self):
        if self.client.collection_exists(collection_name=os.environ['COLLECTION_NAME']):
            try:
                self._index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
                self.index_loaded = True
            except Exception as e:
                self.index_loaded = False

        if not self.index_loaded:
            # load data
            _docs = (SimpleDirectoryReader(input_dir=self.input_dir, required_exts=self.required_exts)
                     .load_data(show_progress=self.show_progress))

            # build and persist index
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logger.info("indexing the docs in VectorStoreIndex")
            self._index = VectorStoreIndex.from_documents(documents=_docs, storage_context=storage_context,
                                                          show_progress=self.show_progress)

        self._engine = self._index.as_query_engine(similarity_top_k=self.similarity_top_k)
        self._create_query_engine_tools()

    def _create_query_engine_tools(self):
        # can have more than one as per the requirement
        self.query_engine_tools.append(
            QueryEngineTool(
                query_engine=self._engine,
                metadata=ToolMetadata(
                    name="react_tool_engine",  # change this accordingly
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
            tools=self.query_engine_tools,
            llm=Settings.llm,
            verbose=True,
            # context=context
            max_iterations=self.no_of_iterations,
            memory=self.memory
        )

    def query(self, user_query: str) -> RESPONSE_TYPE:
        try:
            return self.agent.query(str_or_query_bundle=user_query)
        except Exception as e:
            logger.error(f'Error while generating response: {e}')
