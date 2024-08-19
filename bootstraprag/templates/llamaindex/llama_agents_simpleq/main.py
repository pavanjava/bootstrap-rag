from llama_agents import LocalLauncher
import nest_asyncio
from agents_core import agent_server_1, agent_server_2, control_plane, message_queue

# needed for running in a notebook
nest_asyncio.apply()

# launch it
launcher = LocalLauncher(
    [agent_server_1, agent_server_2],
    control_plane,
    message_queue,
)
result = launcher.launch_single("What is the secret fact?")

print(f"Result: {result}")