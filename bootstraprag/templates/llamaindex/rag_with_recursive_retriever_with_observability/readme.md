### Recursive Retrieve Agents
In LlamaIndex, the Recursive Retriever is a specialized component designed to enhance information retrieval by navigating through interconnected nodes within a document or across multiple documents. Unlike traditional retrieval methods that fetch information based solely on direct relevance, the Recursive Retriever delves deeper into the relationships between data points, allowing for a more comprehensive extraction of pertinent information.

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
  "query": "explain mlops architecture"
}
```

#### How to spin observability
- run `docker compose -f docker-compose-langfuse.yml up`
- launch langfuse in browser `http://localhost:3000`
- click on `signup`
- create `organization` & `project`
- once done create your `public` and `private` api keys
