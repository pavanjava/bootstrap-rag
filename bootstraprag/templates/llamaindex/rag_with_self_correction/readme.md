## Instructions to run the code

- Navigate to the root of the project and run the below command
- `pip install -r requirements.txt`
- open `.env` file update your qdrant url `DB_URL` & password `DB_API_KEY`
- In the data folder place your data preferably any ".pdf" or ".txt"
#### Note: ensure your qdrant and ollama (if LLM models are pointing to local) are running
- run `python main.py`

### Points to observe
- the driver program `main.py` will query the data using 3 different methods as below configure them according to your need.

  - query_with_retry_query_engine(query=user_query)
  - query_with_source_query_engine(query=user_query)
  - query_with_guideline_query_engine(query=user_query)

### How to expose RAG as API
- run `python api_server.py`
- verify the swagger redoc and documentation as below
- open browser and hit `http://localhost:8000/redoc`
- open browser and hit `http://localhost:8000/docs`

### Payload Specification

- Method: POST
- API: http://localhost:8000/api/v1/chat-completion
- Body:
```json
{
  "query": "explain mlops architecture",
  "method": "guideline_query_engine"
}
```
- allowed methods: `source_query_engine`, `guideline_query_engine` and `retry_query_engine` (default)

Note: if you dont pass the method also it works as `retry_query_engine` will be the default engine.