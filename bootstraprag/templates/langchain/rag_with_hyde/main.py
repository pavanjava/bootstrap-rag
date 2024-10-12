import os

from core_advanced_rag import RetrievalAugmentationGenerationUsingHyDE
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

if __name__ == "__main__":
    # Configurations
    FILE_PATH = 'data/mlops.pdf'
    COLLECTION_NAME = os.environ.get('COLLECTION_NAME')
    VECTOR_NAME = os.environ.get('VECTOR_NAME')
    PROMPT_TEMPLATE = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    LLM_MODEL = os.environ.get('LLM_MODEL')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
    QDRANT_URL = os.environ.get('QDRANT_URL')
    QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL')

    # Initialize QnA Pipeline Handler
    pipeline_handler = RetrievalAugmentationGenerationUsingHyDE(
        file_path=FILE_PATH,
        collection_name=COLLECTION_NAME,
        vector_name=VECTOR_NAME,
        prompt_template=PROMPT_TEMPLATE,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        base_url=OLLAMA_BASE_URL
    )

    # Load Documents
    documents = pipeline_handler.load_documents()

    # Get Embeddings
    embeddings = pipeline_handler.get_embeddings()

    # Setup Qdrant and Add Documents
    pipeline_handler.setup_qdrant_collection(embeddings=embeddings, documents=documents)

    # Execute Retrieval Pipeline
    question = "what are the system and operational challenges of mlops?"
    output = pipeline_handler.execute_pipeline(user_query=question)
    print(output)
