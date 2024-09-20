from llama_agents import (
    AgentService,
    AgentOrchestrator,
    ControlPlaneServer,
    SimpleMessageQueue,
)

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from rag_operations import RAGOperations
from dotenv import load_dotenv, find_dotenv
from typing import Any
import os

_ = load_dotenv(find_dotenv())
llm = Ollama(model=os.environ['OLLAMA_LLM_MODEL'], base_url=os.environ['OLLAMA_BASE_URL'], request_timeout=300)
embed_model = OllamaEmbedding(model_name=os.environ['OLLAMA_EMBED_MODEL'], base_url=os.environ['OLLAMA_BASE_URL'])

Settings.llm = llm
Settings.embed_model = embed_model

rag_ops = RAGOperations()


# create an agent
def retrieve_information(query: str) -> Any:
    """used to retrieve the information for user asked questions"""
    return rag_ops.query_engine.query(query)


class SimpleQAgent:
    tool = FunctionTool.from_defaults(fn=retrieve_information)

    agent1 = ReActAgent.from_tools([tool], llm=Settings.llm)

    # create our multi-agent framework components
    message_queue = SimpleMessageQueue(port=8000)
    control_plane = ControlPlaneServer(
        message_queue=message_queue,
        orchestrator=AgentOrchestrator(llm=Settings.llm),
        port=8001,
    )
    agent_server_1 = AgentService(
        agent=agent1,
        message_queue=message_queue,
        description="Useful for getting the answer from user query.",
        service_name="question_answer_agent",
        port=8002,
    )
