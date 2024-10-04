from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.base.response.schema import Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
from llama_index.core.query_engine import RetryQueryEngine, RetrySourceQueryEngine, RetryGuidelineQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator, GuidelineEvaluator
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from rag_evaluator import RAGEvaluator
from dotenv import load_dotenv, find_dotenv
from typing import Union
import qdrant_client
import logging
import os

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class SelfCorrectingRAG:
    RESPONSE_TYPE = Union[
        Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
    ]

    def __init__(self, input_dir: str, similarity_top_k: int = 3, chunk_size: int = 128,
                 chunk_overlap: int = 100, show_progress: bool = False, no_of_retries: int = 5,
                 required_exts: list[str] = ['.pdf', '.txt']):

        self.input_dir = input_dir
        self.similarity_top_k = similarity_top_k
        self.show_progress = show_progress
        self.no_of_retries = no_of_retries
        self.index_loaded = False
        self.required_exts = required_exts

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

        self.rag_evaluator = RAGEvaluator()

        # Create a local Qdrant vector store
        logger.info("initializing the vector store related objects")
        self.client: qdrant_client.QdrantClient = qdrant_client.QdrantClient(url=os.environ['DB_URL'],
                                                                             api_key=os.environ['DB_API_KEY'])
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=os.environ['COLLECTION_NAME'])
        self.query_response_evaluator = RelevancyEvaluator()
        self.base_query_engine = None
        self._index = None

        self._load_data_and_create_engine()

    def _load_data_and_create_engine(self):

        if self.client.collection_exists(collection_name=os.environ['COLLECTION_NAME']):
            try:
                self._index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
                self.base_query_engine = self._index.as_query_engine()
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
            self.base_query_engine = self._index.as_query_engine()

    # The retry query engine uses an evaluator to improve the response from a base query engine.
    #
    # It does the following:
    #
    # first queries the base query engine, then
    # use the evaluator to decided if the response passes.
    # If the response passes, then return response,
    # Otherwise, transform the original query with the evaluation result (query, response, and feedback)
    # into a new query, Repeat up to max_retries
    def query_with_retry_query_engine(self, query: str) -> RESPONSE_TYPE:

        retry_query_engine = RetryQueryEngine(self.base_query_engine, self.query_response_evaluator,
                                              max_retries=self.no_of_retries)
        retry_response = retry_query_engine.query(query)
        return retry_response

    # The Source Retry modifies the query source nodes by filtering the existing
    # source nodes for the query based on llm node evaluation.
    def query_with_source_query_engine(self, query: str) -> RESPONSE_TYPE:
        retry_source_query_engine = RetrySourceQueryEngine(self.base_query_engine,
                                                           self.query_response_evaluator,
                                                           max_retries=self.no_of_retries)
        retry_source_response = retry_source_query_engine.query(query)
        return retry_source_response

    # This module tries to use guidelines to direct the evaluator's behavior.
    # You can customize your own guidelines.
    def query_with_guideline_query_engine(self, query: str) -> RESPONSE_TYPE:
        # Guideline eval
        guideline_eval = GuidelineEvaluator(
            guidelines=DEFAULT_GUIDELINES + "\nThe response should not be overly long.\n"
                                            "The response should try to summarize where possible.\n"
        )  # just for example
        retry_guideline_query_engine = RetryGuidelineQueryEngine(self.base_query_engine,
                                                                 guideline_eval, resynthesize_query=True,
                                                                 max_retries=self.no_of_retries)
        retry_guideline_response = retry_guideline_query_engine.query(query)
        if os.environ.get('IS_EVALUATION_NEEDED') == 'true':
            self.rag_evaluator.evaluate(user_query=query, response_obj=retry_guideline_response)
        return retry_guideline_response
