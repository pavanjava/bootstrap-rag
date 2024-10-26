## Instructions to run the code

#### How to spin observability
- run `docker compose -f docker-compose-langfuse.yml up`
- launch langfuse in browser `http://localhost:3000`
- click on `signup`
- create `organization` & `project`
- once done create your `public` and `private` api keys

#### Running the project
- Navigate to the root of the project and run the below command
- `pip install -r requirements.txt`
- open `.env` file update your qdrant password in the property `DB_API_KEY`
- In the data folder place your data preferably any ".pdf"
#### Note: ensure your qdrant and ollama (if LLM models are pointing to local) are running
- run `python main.py`

Note: This is Human in the loop agent, so keep a watch on the console to pass the human feedback to the agent.
