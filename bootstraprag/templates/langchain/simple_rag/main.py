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


class MLOpsQA:
    def __init__(self, file_path: str, collection_name: str, qdrant_url: str, qdrant_api_key: str):
        self.file_path = file_path
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        self.model = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")
        self.embedding = FastEmbedEmbeddings(model='snowflake/snowflake-arctic-embed-s')
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        self.setup_qdrant()
        self.documents = self.load_and_split_documents()
        self.vector_store = self.setup_vector_store()
        self.retrieval_chain = self.setup_retrieval_chain()

    def setup_qdrant(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "content": VectorParams(size=384, distance=Distance.COSINE)
                }
            )

    def load_and_split_documents(self) -> List[Any]:
        loader = PyMuPDFLoader(file_path=self.file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        return loader.load_and_split(text_splitter=text_splitter)

    def insert_data_with_metadata(self):
        chunked_data = []

        for doc in self.documents:
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
        template = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        Question: {input}
        Context: {context}

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template=template)
        retriever = self.vector_store.as_retriever()
        combine_docs_chain = create_stuff_documents_chain(self.model, prompt)
        return create_retrieval_chain(retriever, combine_docs_chain)

    def answer_question(self, question: str) -> str:
        result = self.retrieval_chain.invoke({"input": question})
        return result["answer"]


# Usage example:
if __name__ == "__main__":
    mlops_qa = MLOpsQA(
        file_path='data/mlops.pdf',
        collection_name="test_langchain_collection",
        qdrant_url="http://localhost:6333",
        qdrant_api_key='th3s3cr3tk3y'
    )

    # Uncomment the following line to insert data (only needed once)
    # mlops_qa.insert_data_with_metadata()

    question = "What are the challenges of MLOps?"
    answer = mlops_qa.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
