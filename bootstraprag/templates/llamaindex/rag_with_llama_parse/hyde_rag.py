import os

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer)
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
# enable if you are using openai
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
# enable if you are using openai
# from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.base.response.schema import Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
from llama_parse import LlamaParse
import qdrant_client
import logging
from dotenv import load_dotenv, find_dotenv
from typing import Union

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class RAGWithHyDeEngine:
    RESPONSE_TYPE = Union[
        Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
    ]

    def __init__(self, data_path: str, chunk_size: int = 512, chunk_overlap: int = 200,
                 similarity_top_k: int = 3):
        # load the local data directory and chunk the data for further processing
        self.docs = self._docs_with_llama_parse(data_path=data_path)
        self.text_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Create a local Qdrant vector store
        logger.info("initializing the vector store related objects")
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

        self.text_chunks = []
        self.doc_ids = []
        self.nodes = []

        self.similarity_top_k = similarity_top_k
        self.hyde_query_engine: TransformQueryEngine = None

        # preprocess the data like chunking, nodes, metadata etc
        self._pre_process()

    def _docs_with_llama_parse(self, data_path: str, ):
        # set up parser
        parser = LlamaParse(
            result_type="markdown",  # "markdown" and "text" are available
            api_key=os.environ.get('llama_cloud')
        )

        # use SimpleDirectoryReader to parse our file
        file_extractor = {"pdf": parser}
        documents = SimpleDirectoryReader(input_dir=data_path, file_extractor=file_extractor).load_data(
            show_progress=True)
        return documents

    def _pre_process(self):
        logger.info("enumerating docs")
        for doc_idx, doc in enumerate(self.docs):
            curr_text_chunks = self.text_parser.split_text(doc.text)
            self.text_chunks.extend(curr_text_chunks)
            self.doc_ids.extend([doc_idx] * len(curr_text_chunks))

        logger.info("enumerating text_chunks")
        for idx, text_chunk in enumerate(self.text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = self.docs[self.doc_ids[idx]]
            node.metadata = src_doc.metadata
            self.nodes.append(node)

        logger.info("enumerating nodes")
        for node in self.nodes:
            node_embedding = Settings.embed_model.get_text_embedding(
                node.get_content(metadata_mode=MetadataMode.ALL)
            )
            node.embedding = node_embedding

        # create vector store, index documents and creates retriever
        self._create_index_and_retriever()

    def _create_index_and_retriever(self):
        logger.info("initializing the storage context")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        logger.info("indexing the nodes in VectorStoreIndex")
        if not self.client.collection_exists(collection_name=os.environ['COLLECTION_NAME']):
            index = VectorStoreIndex(
                nodes=self.nodes,
                storage_context=storage_context,
                transformations=Settings.transformations,
            )
        else:
            index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        logger.info("initializing the VectorIndexRetriever with top_k as 5")
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=self.similarity_top_k)
        response_synthesizer = get_response_synthesizer()
        logger.info("creating the RetrieverQueryEngine instance")
        vector_query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            response_synthesizer=response_synthesizer,
        )
        logger.info("creating the HyDEQueryTransform instance")
        hyde = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(vector_query_engine, hyde)

        self.hyde_query_engine = hyde_query_engine

    def query(self, query_string: str) -> RESPONSE_TYPE:
        try:
            response = self.hyde_query_engine.query(str_or_query_bundle=query_string)
            return response
        except Exception as e:
            logger.error(f'Error while inference: {e}')
