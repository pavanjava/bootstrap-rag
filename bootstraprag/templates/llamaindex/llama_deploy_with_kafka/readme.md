## llama-agents

### install qdrant
- `docker pull qdrant/qdrant`
- `docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant`

### install ollama
- navigate to [https://ollama.com/download](https://ollama.com/download)

### how to run llama-agents
- open `.env`
- change the `DB_URL`, `DB_API_KEY` and `COLLECTION_NAME` according to you
- point the right right LLMs (if not local)
- `pip install -r requirements.txt`
- `python main.py`
