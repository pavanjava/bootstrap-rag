## llama-workflows with llama-deploy

### install Required Software (Ollama and Qdrant)
- follow the documentation from the root folder

### how to run llama-agents
- open `.env`
- change the `DB_URL`, `DB_API_KEY` and `COLLECTION_NAME` according to you
- point the right LLMs (if not local)
- `pip install -r requirements.txt`
- `python deploy_code.py`
- `python deploy_workflow.py`
- `python main.py`
