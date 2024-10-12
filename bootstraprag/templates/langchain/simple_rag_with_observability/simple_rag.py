import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaLLM
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import Any, List
from uuid import uuid4
from dotenv import load_dotenv, find_dotenv
from custom_templates import chat_prompt_template
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import phoenix as px

px.launch_app()
tracer_provider = register(
    project_name="simple-rag",
    endpoint="http://127.0.0.1:4317",  # change this to remote if needed
    set_global_tracer_provider=True

)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)


class SimpleRAG:
    def __init__(self, file_path: str, collection_name: str, qdrant_url: str, qdrant_api_key: str):
        load_dotenv(find_dotenv())
        self.file_path = file_path
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        self.model = OllamaLLM(model=os.environ.get("OLLAMA_LLM_MODEL"), base_url=os.environ.get("OLLAMA_BASE_URL"))
        self.embedding = FastEmbedEmbeddings(model=os.environ.get("EMBEDDING_MODEL"))
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        # self.documents = self.load_and_split_documents()
        self.setup_qdrant()
        self.vector_store = self.setup_vector_store()
        self.retrieval_chain = self.setup_retrieval_chain()

    def setup_qdrant(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "content": VectorParams(size=384, distance=Distance.COSINE)
                    }
                )

                self.insert_data_with_metadata()
            except Exception as e:
                print(f"Exception: {str(e)}")

    def load_and_split_documents(self) -> List[Any]:
        loader = PyMuPDFLoader(file_path=self.file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        return loader.load_and_split(text_splitter=text_splitter)

    def insert_data_with_metadata(self):
        documents = self.load_and_split_documents()
        chunked_data = []

        for doc in documents:
            id = str(uuid4())
            content = doc.page_content
            source = doc.metadata['source']
            page = doc.metadata['page']

            content_vector = self.embedding.embed_documents([content])[0]
            vector_dict = {"content": content_vector}

            payload = {
                "page_content": content,
                "metadata": {
                    "id": id,
                    "page_content": content,
                    "source": source,
                    "page": page,
                }
            }

            metadata = PointStruct(id=id, vector=vector_dict, payload=payload)
            chunked_data.append(metadata)

        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=chunked_data)

    def setup_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(client=self.client, collection_name=self.collection_name, embedding=self.embedding,
                                 vector_name="content")

    def setup_retrieval_chain(self):
        prompt = ChatPromptTemplate.from_template(template=chat_prompt_template)
        retriever = self.vector_store.as_retriever()
        combine_docs_chain = create_stuff_documents_chain(self.model, prompt)
        return create_retrieval_chain(retriever, combine_docs_chain)

    def query(self, user_query: str) -> str:
        result = self.retrieval_chain.invoke({"input": user_query})
        return result["answer"]
