from langchain_core.output_parsers import StrOutputParser
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from qdrant_client.http.models import Distance, VectorParams
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import phoenix as px
import qdrant_client

px.launch_app()
tracer_provider = register(
    project_name="rag-with-hyde",
    endpoint="http://127.0.0.1:4317",  # change this to remote if needed
    set_global_tracer_provider=True

)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)


class RetrievalAugmentationGenerationUsingHyDE:
    def __init__(self, file_path, collection_name, vector_name, prompt_template, llm_model, embedding_model, qdrant_url,
                 qdrant_api_key, base_url):
        self.file_path = file_path
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.prompt_template = prompt_template
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.llm = ChatOllama(model=llm_model, temperature=0.2, base_url=base_url)
        self.base_embeddings = OllamaEmbeddings(model=embedding_model)
        self.client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.vector_store = None

    def load_documents(self):
        loader = PyMuPDFLoader(file_path=self.file_path)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=30
        )
        return loader.load_and_split(text_splitter=text_splitter)

    def get_embeddings(self):
        prompt = PromptTemplate(input_variables=["question"], template=self.prompt_template)
        llm_chain = self.llm | prompt
        return HypotheticalDocumentEmbedder(
            llm_chain=llm_chain,
            base_embeddings=self.base_embeddings
        ).from_llm(llm=self.llm, base_embeddings=self.base_embeddings, prompt_key="web_search")

    def setup_qdrant_collection(self, embeddings, documents):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "content": VectorParams(size=384, distance=Distance.COSINE)
                }
            )
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=embeddings,
                vector_name=self.vector_name
            )
            self.vector_store.add_documents(documents=documents)
        else:
            self.vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                vector_name=self.vector_name,
                embedding=embeddings
            )

    def execute_pipeline(self, user_query):
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.8}
        )
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        return chain.invoke(user_query)
