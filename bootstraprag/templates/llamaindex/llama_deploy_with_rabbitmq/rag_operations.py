from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
# enable if you are using openai
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
# enable if you are using openai
# from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from typing import Union
import qdrant_client
import logging
import os

_ = load_dotenv(find_dotenv())
logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class RAGOperations:

    def __init__(self, data_path: str = './data', show_progress: bool = True, chunk_size: int = 512,
                 chunk_overlap: int = 200):
        self.documents = SimpleDirectoryReader(input_dir=data_path).load_data(show_progress=show_progress)
        self.text_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.client = qdrant_client.QdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=os.environ['COLLECTION_NAME'])

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

        Settings.transformations = [self.text_parser]
        self.query_engine: BaseQueryEngine = None

        self._index_and_create_query_engine()

    def _index_and_create_query_engine(self):
        logger.info("initializing the storage context")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        logger.info("indexing the nodes in VectorStoreIndex")
        if not self.client.collection_exists(collection_name=os.environ['COLLECTION_NAME']):
            vector_index = VectorStoreIndex.from_documents(
                documents=self.documents,
                storage_context=storage_context,
                transformations=Settings.transformations,
            )
        else:
            vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        self.query_engine = vector_index.as_query_engine()
