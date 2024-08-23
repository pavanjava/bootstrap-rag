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


llm = Ollama(model=os.environ['OLLAMA_LLM_MODEL'], base_url=os.environ['OLLAMA_BASE_URL'], request_timeout=300)
embed_model = OllamaEmbedding(model_name=os.environ['OLLAMA_EMBED_MODEL'], base_url=os.environ['OLLAMA_BASE_URL'])

Settings.llm = llm
Settings.embed_model = embed_model


# create an agent
def get_the_secret_fact() -> str:
    """Returns the secret fact."""
    return "The secret fact is: A baby llama is called a 'Cria'."

class SimpleQAgent:
    tool = FunctionTool.from_defaults(fn=get_the_secret_fact)

    agent1 = ReActAgent.from_tools([tool], llm=Settings.llm)
    agent2 = ReActAgent.from_tools([], llm=Settings.llm)

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
        description="Useful for getting the secret fact.",
        service_name="secret_fact_agent",
        port=8002,
    )
    agent_server_2 = AgentService(
        agent=agent2,
        message_queue=message_queue,
        description="Useful for getting random dumb facts.",
        service_name="dumb_fact_agent",
        port=8003,
    )
