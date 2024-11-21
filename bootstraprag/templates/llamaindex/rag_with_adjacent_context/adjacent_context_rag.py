import os

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.postprocessor import PrevNextNodePostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from typing import List, Any
from dotenv import load_dotenv, find_dotenv
import qdrant_client


class AdjacentContextRAG:
    def __init__(self, exts: List[str] = ("pdf", "text"), is_show_progress: bool = True, chunk_size: int = 256,
                 chunk_overlap: int = 20):
        load_dotenv(find_dotenv())
        self.documents = SimpleDirectoryReader(input_dir="./data", required_exts=exts).load_data(
            show_progress=is_show_progress)
        llm = Ollama(model=os.environ.get("OLLAMA_LLM_MODEL"), base_url=os.environ.get("OLLAMA_BASE_URL"))
        embedding = OllamaEmbedding(model_name=os.environ.get("OLLAMA_EMBED_MODEL"),
                                    base_url=os.environ.get("OLLAMA_BASE_URL"))
        self.qdrant_client = qdrant_client.QdrantClient(url=os.environ.get("DB_URL"), api_key=os.environ.get("DB_API_KEY"))

        Settings.llm = llm
        Settings.embed_model = embedding
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        self.vector_index: VectorStoreIndex = None
        self.query_engine: BaseQueryEngine = None
        self._init_vector_index()

    def _init_vector_index(self):
        vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=os.environ.get("COLLECTION_NAME"))
        if not self.qdrant_client.collection_exists(collection_name=os.environ.get("COLLECTION_NAME")):
            # build and persist index
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.vector_index = VectorStoreIndex.from_documents(storage_context=storage_context, documents=self.documents)
        else:
            self.vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    def _create_query_engine(self):
        node_postprocessor = PrevNextNodePostprocessor(docstore=self.vector_index.docstore, num_nodes=5)
        self.query_engine = self.vector_index.as_query_engine(
            similarity_top_k=1,
            node_postprocessors=[node_postprocessor],
            response_mode="tree_summarize",
        )

    def query(self, query: str = ""):
        self.query_engine.query(str_or_query_bundle=query)



