### Introspective Financial Agents

Introspective agents are artificial intelligence systems designed to evaluate and refine their own outputs through iterative reflection and correction. This self-assessment process enables them to improve performance over time. For instance, in the LlamaIndex framework, an introspective agent generates an initial response to a task and then engages in reflection and correction cycles to enhance the response until a satisfactory outcome is achieved.

An introspective worker agent is a specific implementation of this concept. It operates by delegating tasks to two other agents: one generates the initial response, and the other performs reflection and correction on that response. This structured approach ensures systematic evaluation and improvement of outputs.

### Setup Langfuse for observability
- `docker-compose run -f docker-compose-langfuse.yml`

### How to run ?
- `pip install -r requirements.txt`
- `python main.py` [standalone code]
- `python api_server.py` [litserve apis]

### How to access API?
- URL: http://localhost:8002/predict
- method: POST
- payload: 
```json
{
"query": "I have 10k dollars Now analyze APPL stock and MSFT stock to let me know where to invest this money and how many stock will I get it, Give me the last 3 months historical close prices of APPL and MSFT. Respond with a comparative summary on closing prices and recommended stock to invest."
}
```

### How to check traces?
launch `http://localhost:3000`
