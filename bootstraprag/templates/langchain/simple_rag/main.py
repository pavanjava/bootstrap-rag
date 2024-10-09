from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

embeddings = FastEmbedEmbeddings()

client = QdrantClient(url='http://localhost:6333', api_key='th3s3cr3tk3y')

client.create_collection(
    collection_name="demo_collection_1",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection_1",
    embedding=embeddings
)

WEBSITE_URL = "https://medium.com/@jaintarun7/multimodal-using-gemini-and-llamaindex-f622a190cc32"
data = WebBaseLoader(WEBSITE_URL)
docs = data.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)
chunks = text_split.split_documents(docs)

qdrant = vector_store.from_documents(chunks, embedding=embeddings, api_key='th3s3cr3tk3y')
