from llama_agents import LocalLauncher
import nest_asyncio
from agents_core import SimpleQAgent


simple_agent = SimpleQAgent()

# needed for running in a notebook
nest_asyncio.apply()

# launch it
launcher = LocalLauncher(
    [simple_agent.agent_server_1, simple_agent.agent_server_2],
    simple_agent.control_plane,
    simple_agent.message_queue,
)
result = launcher.launch_single("What is the secret fact?")

print(f"Result: {result}")