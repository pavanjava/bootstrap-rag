# driver code
from controllable_agents_with_human_in_the_loop import ControllableAgentsWithHumanInLoop


controllable_agent = ControllableAgentsWithHumanInLoop(input_dir='data', show_progress=True)

# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query [type 'bye' to 'exit']: ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye" or user_query.lower() == "exit":
        break

    response = controllable_agent.chat_repl(user_query=user_query)
    print(response)