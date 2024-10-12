# Advanced RAG with HyDE Project

This project implements a Advanced RAG with HyDE based Question-Answering system using LangChain, Ollama, and Qdrant.

## Prerequisites

- Python 3.8 or higher
- Ollama running locally (for LLM)
- Qdrant running locally (for vector storage)

## project structure
```tree
.
├── Dockerfile
├── __init__.py
├── api_server.py
├── client.py
├── core_advanced_rag.py
├── data
│   └── mlops.pdf
├── main.py
├── readme.md
└── requirements.txt
```

## Installation

1. `pip install bootstrap-rag`

### Setting up Ollama and Qdrant
Method 1:
1. navigate to root_folder/setups
2. run the docker-compose-dev.yml
3. run the pull_model as per the underlying OS

Method 2:
1. Install and run Ollama:
    - Follow the instructions at [Ollama's official website](https://ollama.ai/) to install Ollama.
    - Make sure Ollama is running and accessible at `http://localhost:11434`.

2. Install and run Qdrant:
    - Follow the instructions at [Qdrant's official website](https://qdrant.tech/documentation/quick-start/) to install Qdrant.
    - Make sure Qdrant is running and accessible at `http://localhost:6333`.

## How to Run
1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. run `bootstraprag create <your_poc_project_name>`

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Prepare your MLOps PDF document and place it in the `data` directory.

2. Update the `.env` file with your specific configuration:
    - Update the `file_path` to point to your PDF document.
    - Update the `collection_name` if you want to use a different name for your Qdrant collection.
    - Update the `qdrant_url` and `qdrant_api_key` if your Qdrant setup is different.

3. Run the script:
   ```
   python main.py
   ```
   or
   ```
   python api_server.py
   ```