from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.base.response.schema import Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
from dotenv import load_dotenv, find_dotenv
from rag_evaluator import RAGEvaluator
from typing import Union
import qdrant_client
import logging
import os

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class SimpleRAG:
    RESPONSE_TYPE = Union[
        Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
    ]

    def __init__(self, input_dir: str, similarity_top_k: int = 3, chunk_size: int = 128, chunk_overlap: int = 100,
                 show_progress: bool = False):
        self.index_loaded = False
        self.similarity_top_k = similarity_top_k
        self.input_dir = input_dir
        self._index: VectorStoreIndex = None
        self._engine = None
        self.agent: ReActAgent = None
        self.query_engine_tools = []
        self.show_progress = show_progress

        self.rag_evaluator = RAGEvaluator()

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
        self.client: qdrant_client.QdrantClient = qdrant_client.QdrantClient(url=os.environ['DB_URL'],
                                                                             api_key=os.environ['DB_API_KEY'])
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=os.environ['COLLECTION_NAME'])

        self._create_index()

    def _create_index(self):

        if self.client.collection_exists(collection_name=os.environ['COLLECTION_NAME']):
            try:
                self._index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
                self.index_loaded = True
            except Exception as e:
                self.index_loaded = False

        if not self.index_loaded:
            # load data
            _docs = SimpleDirectoryReader(input_dir=self.input_dir).load_data(show_progress=self.show_progress)

            # build and persist index
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logger.info("indexing the docs in VectorStoreIndex")
            self._index = VectorStoreIndex.from_documents(documents=_docs,
                                                          storage_context=storage_context,
                                                          show_progress=self.show_progress)

    def do_rag(self, user_query: str) -> RESPONSE_TYPE:

        logger.info("retrieving the relavent nodes")
        query_engine = self._index.as_query_engine(similarity_top_k=self.similarity_top_k)
        logger.info("LLM is thinking...")
        response = query_engine.query(str_or_query_bundle=user_query)
        logger.info(f'response: {response}')
        if os.environ.get('IS_EVALUATION_NEEDED') == 'true':
            self.rag_evaluator.evaluate(user_query=user_query, response_obj=response)
        return response
