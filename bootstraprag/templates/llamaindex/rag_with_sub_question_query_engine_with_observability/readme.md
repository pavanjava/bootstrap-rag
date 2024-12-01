### Sub Question Query Engine
The Sub-Question Query Engine in LlamaIndex is designed to handle complex queries that require information from multiple data sources. It operates by decomposing a complex query into several sub-questions, each directed to the most relevant data source. After obtaining responses to these sub-questions, it synthesizes them into a comprehensive final answer.

### How to run?
`pip install -r requirements.txt`
`python main.py`

### How to expose as API?
`python api_server.py`
- Method: POST
- API: http://localhost:8000/api/v1/chat-completion
- Body:
```json
{
  "query": "Explain vertical plane in orthodontics"
}
```

#### How to spin observability
- run `docker compose -f docker-compose-langfuse.yml up`
- launch langfuse in browser `http://localhost:3000`
- click on `signup`
- create `organization` & `project`
- once done create your `public` and `private` api keys
