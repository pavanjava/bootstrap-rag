# LLM as Judge (scoped to CRAG)

This project implements a LLM as Judge concept to measure 
- answer_hallucination
- generation_hallucination
- retrieval_hallucination

eventually this project will be converted as CRAG project.

## Prerequisites

- Python 3.8 or higher
- Ollama running locally (for LLM)
- Qdrant running locally (for vector storage)

### Project structure
```.
├── Dockerfile
├── __init__.py
├── api_server.py
├── custom_templates.py
├── data
│   └── mlops.pdf
├── llm_as_judge.py
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
1. Create a virtual environment (optional but recommended): `python -m venv venv`
2. `source venv/bin/activate`  # On Windows, use `venv\Scripts\activate`
3. run `bootstraprag create <your_poc_project_name>`
4. Install the required dependencies: `pip install -r requirements.txt`
5. run `python main.py` or `python api_server.py`


