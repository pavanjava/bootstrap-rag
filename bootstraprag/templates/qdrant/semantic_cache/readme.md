## Qdrant Semantic Cache
Semantic Cache is a superfast cache mechanism on contextual meaning very much useful for LLM giving same response with out much deviation.

### How to run
- `pip install -r requirements.txt`
- `python semantic_cache.py`

### Expose Semantic Cache as API
- `python api_server.py`
```text
API: http://localhost:8000/api/v1/chat-completion
Method: POST
payload: {
  "question": "what is the capital of India?"
}
```
