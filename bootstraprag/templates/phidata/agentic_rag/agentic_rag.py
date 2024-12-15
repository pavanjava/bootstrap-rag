import os
import typer
import qdrant_client
from typing import Optional
from rich.prompt import Prompt

from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase, PDFUrlKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = os.getenv("collection_name")

qdrantClient = qdrant_client.QdrantClient(url=qdrant_url, api_key=api_key)

vector_db = Qdrant(
    collection=collection_name,
    embedder=OllamaEmbedder(model="nomic-embed-text:latest", dimensions=768, host="http://localhost:11434"),
    url=qdrant_url,
    api_key=api_key,
)

knowledge_base = PDFKnowledgeBase(
    path="data",
    vector_db=vector_db,
)

# Comment out after first run
if not qdrantClient.collection_exists(collection_name=collection_name):
    knowledge_base.load(recreate=True, upsert=True, skip_existing=True)


def qdrant_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        run_id=run_id,
        model=Ollama(id="llama3.2:latest", host="http://localhost:11434"),
        user_id=user,
        knowledge=knowledge_base,
        tool_calls=True,
        use_tools=True,
        show_tool_calls=True,
        debug_mode=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    return agent


# if __name__ == "__main__":
#     _agent = qdrant_agent('pavan_mantha')
#     user_id = 'pavan_mantha' # change the user id accordingly
#     while True:
#         message = Prompt.ask(f"[bold] :sunglasses: {user_id} [/bold]")
#         if message in ("exit", "bye"):
#             break
#         # _agent.print_response(message)
#         resp = _agent.run(message=message)
#         print(resp)
