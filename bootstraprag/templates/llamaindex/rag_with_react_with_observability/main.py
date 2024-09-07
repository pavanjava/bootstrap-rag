# driver code
from react_agent_with_query_engine import ReActWithQueryEngine


react_with_engine = ReActWithQueryEngine(input_dir='data', show_progress=True)

# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query [type 'bye' to 'exit']: ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye" or user_query.lower() == "exit":
        break

    response = react_with_engine.query(user_query=user_query)
    print(response)