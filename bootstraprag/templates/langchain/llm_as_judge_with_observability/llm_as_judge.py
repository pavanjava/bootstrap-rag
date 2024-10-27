import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.utils import Output
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv
from qdrant_client.http.models import VectorParams, Distance
from typing import List, Any
from custom_templates import (
    retrieval_grader_template,
    hallucination_grading_template,
    answer_generating_template,
    answer_grading_template
)
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import phoenix as px

px.launch_app()
tracer_provider = register(
    project_name="llm-as-judge",
    endpoint="http://127.0.0.1:4317",  # change this to remote if needed
    set_global_tracer_provider=True

)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)


class LLMasJudge:
    def __init__(self, file_path: str, collection_name: str, qdrant_url: str, qdrant_api_key: str):
        load_dotenv(find_dotenv())
        self.file_path = file_path
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        self.model = OllamaLLM(model=os.environ.get("OLLAMA_LLM_MODEL"), base_url=os.environ.get("OLLAMA_BASE_URL"))
        self.embedding = FastEmbedEmbeddings(model=os.environ.get("EMBEDDING_MODEL"))
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        # LLM
        self.llm = ChatOllama(model=os.environ.get('OLLAMA_LLM_MODEL'), format="json")
        self.vector_store: QdrantVectorStore = None
        self.documents = self.load_and_split_documents()
        self.setup_qdrant()

    def load_and_split_documents(self) -> List[Any]:
        loader = PyMuPDFLoader(file_path=self.file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        return loader.load_and_split(text_splitter=text_splitter)

    def setup_qdrant(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "content": VectorParams(size=384, distance=Distance.COSINE)
                    }
                )
                self.load_data_to_qdrant()
            except Exception as e:
                print(f"Exception: {str(e)}")
        else:
            self.vector_store = QdrantVectorStore.from_existing_collection(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                embedding=self.embedding,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="content"
            )

    def load_data_to_qdrant(self):
        vector_store: QdrantVectorStore = QdrantVectorStore(client=self.client, collection_name=self.collection_name,
                                                            embedding=self.embedding, vector_name="content",
                                                            retrieval_mode=RetrievalMode.DENSE)
        vector_store.add_documents(
            documents=self.documents
        )
        self.vector_store = vector_store

    def retrieval_grader(self, question: str):
        prompt = PromptTemplate(
            template=retrieval_grader_template,
            input_variables=["question", "document"],
        )
        retrieval_grader = prompt | self.llm | JsonOutputParser()
        docs = self.vector_store.as_retriever().invoke(question)
        doc_txt = docs[1].page_content
        retrieval_grading_response = retrieval_grader.invoke({"question": question, "document": doc_txt})
        return retrieval_grading_response

    def generate(self, question: str) -> Output:
        prompt = PromptTemplate(
            template=answer_generating_template,
            input_variables=["question", "context"]
        )

        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()

        # Run
        docs = self.vector_store.as_retriever().invoke(question)
        generation: Output = rag_chain.invoke({"context": docs, "question": question})
        return generation

    def hallucination_grader(self, question: str, generation):
        prompt = PromptTemplate(
            template=hallucination_grading_template,
            input_variables=["generation", "documents"],
        )
        docs = self.vector_store.as_retriever().invoke(question)
        hallucination_grader = prompt | self.llm | JsonOutputParser()
        hallucination_grading_response = hallucination_grader.invoke({"documents": docs, "generation": generation})
        return hallucination_grading_response

    def answer_grader(self, question: str, generation: str):
        prompt = PromptTemplate(
            template=answer_grading_template,
            input_variables=["generation", "question"]
        )
        answer_grader = prompt | self.llm | JsonOutputParser()
        answer_grading_response = answer_grader.invoke({"question": question, "generation": generation})
        return answer_grading_response
