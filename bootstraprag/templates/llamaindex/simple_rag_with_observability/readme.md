## Instructions to run the code

- Navigate to the root of the project and run the below command
- `pip install -r requirements.txt`
- open `.env` file update your qdrant password in the property `DB_API_KEY`
- In the data folder place your data preferably any ".pdf"
#### Note: ensure your qdrant and ollama (if LLM models are pointing to local) are running
- run `python main.py`
- This code is enabled with Observability powered by `Arize Phoenix`.

### How to expose RAG as API
- run `python api_server.py`
- verify the swagger redoc and documentation as below
- open browser and hit `http://localhost:8000/redoc`
- open browser and hit `http://localhost:8000/docs`