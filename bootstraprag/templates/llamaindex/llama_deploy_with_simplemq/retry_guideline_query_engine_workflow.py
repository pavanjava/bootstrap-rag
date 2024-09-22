import os
import logging
import qdrant_client
from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine import RetryGuidelineQueryEngine
from llama_index.core import (VectorStoreIndex, Settings, StorageContext, SimpleDirectoryReader)
from llama_index.core.evaluation import GuidelineEvaluator
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from dotenv import load_dotenv, find_dotenv
from events import QueryEngineEvent

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class RAGWorkflowWithRetryGuidelineQueryEngine(Workflow):
    def __init__(self, index: VectorStoreIndex, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index: VectorStoreIndex = index

    @step
    async def create_retry_query_engine(self, ctx: Context, ev: StartEvent) -> QueryEngineEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        logger.info(f"creating query engine for query: {ev.get('query')}")
        query = ev.get("query")
        no_of_retries = ev.get("no_of_retries", default=3)

        if not query:
            raise ValueError("Query is required!")

        # store the settings in the global context
        await ctx.set("query", query)
        await ctx.set("no_of_retries", no_of_retries)

        base_query_engine = self.index.as_query_engine(llm=Settings.llm, similarity_top_k=2, sparse_top_k=12,
                                                       vector_store_query_mode="hybrid")
        return QueryEngineEvent(base_query_engine=base_query_engine)

    @step
    async def query_with_retry_source_query_engine(self, ctx: Context, ev: QueryEngineEvent) -> StopEvent:
        """Return a response using reranked nodes."""
        query = await ctx.get("query")
        no_of_retries = await ctx.get("no_of_retries")
        base_query_engine: BaseQueryEngine = ev.base_query_engine

        # Guideline eval
        guideline_eval = GuidelineEvaluator(
            guidelines=DEFAULT_GUIDELINES + "\nThe response should not be overly long.\n"
                                            "The response should try to summarize where possible.\n"
        )  # just for example
        retry_guideline_query_engine = RetryGuidelineQueryEngine(base_query_engine, guideline_eval,
                                                                 resynthesize_query=True, max_retries=no_of_retries)
        retry_guideline_response = retry_guideline_query_engine.query(query)
        logger.info(f"response for query is: {retry_guideline_response}")
        return StopEvent(result=str(retry_guideline_response))


def build_rag_workflow_with_retry_guideline_query_engine() -> RAGWorkflowWithRetryGuidelineQueryEngine:
    index_loaded = False
    # host points to qdrant in docker-compose.yml
    client = qdrant_client.QdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])
    aclient = qdrant_client.AsyncQdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])
    vector_store = QdrantVectorStore(collection_name=os.environ['COLLECTION_NAME'], client=client, aclient=aclient,
                                     enable_hybrid=True, batch_size=50)

    Settings.llm = Ollama(model=os.environ['OLLAMA_LLM_MODEL'], base_url=os.environ['OLLAMA_BASE_URL'],
                          request_timeout=600)
    Settings.embed_model = OllamaEmbedding(model_name=os.environ['OLLAMA_EMBED_MODEL'],
                                           base_url=os.environ['OLLAMA_BASE_URL'])

    # index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)
    index: VectorStoreIndex = None

    if client.collection_exists(collection_name=os.environ['COLLECTION_NAME']):
        try:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            index_loaded = True
        except Exception as e:
            index_loaded = False

    if not index_loaded:
        # load data
        _docs = (SimpleDirectoryReader(input_dir='data', required_exts=['.pdf']).load_data(show_progress=True))

        # build and persist index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("indexing the docs in VectorStoreIndex")
        index = VectorStoreIndex.from_documents(documents=_docs, storage_context=storage_context, show_progress=True)

    return RAGWorkflowWithRetryGuidelineQueryEngine(index=index, timeout=120.0)
