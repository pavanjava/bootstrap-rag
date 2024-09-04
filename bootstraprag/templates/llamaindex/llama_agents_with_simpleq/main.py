from llama_agents import LocalLauncher
import nest_asyncio
from agents_core import SimpleQAgent


simple_agent = SimpleQAgent()

# needed for running in a notebook
nest_asyncio.apply()

# launch it
launcher = LocalLauncher(
    [simple_agent.agent_server_1],
    simple_agent.control_plane,
    simple_agent.message_queue,
)

# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query [type 'bye' to 'exit']: ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye" or user_query.lower() == "exit":
        break

    result = launcher.launch_single(user_query)
    print(result)