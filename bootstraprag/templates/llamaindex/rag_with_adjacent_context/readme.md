
### Project Structure

```
.
├── adjacent_context_rag.py   # Implementation of the PrevNextPostprocessorDemo class
├── api_server.py
├── main.py                   # Example usage script
├── data/                     # Directory for storing input documents
├── .env                      # Environment variables for Ollama settings
└── README.md                 # Instructions to run the project
```
### how to run ?

- `pip install -r requirementx.txt`
- edit `.env` file accordingly
- `python main.py`
### want to expose as API ?
- `python api_server.py`
- URI: http://localhost:8000/api/v1/chat/completion
- method: POST
- payload
```json
{
  "query": "what are the problems of mlops"
}
```

