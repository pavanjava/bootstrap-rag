## Semantic Routing
Semantic Router is a superfast decision-making layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic of semantic vector space to make those decisions — routing our requests using semantic meaning.

### How to execute code
1. `pip install -r requirements.txt`
2. `python main.py`

### Expose Semantic Router as API
- `python api_server.py`
```text
API: http://localhost:8000/api/v1/chat-completion
Method: POST
payload: {
  "question": "what is the Weather today?"
}
```