from recursive_retriever_agents_core import RecursiveAgentManager


agent_names = ['mlops', 'attention', 'orthodontics']
agent_manager = RecursiveAgentManager(agent_names)

# Usage example:
if __name__ == "__main__":
    while True:
        # Get user input
        user_query = input("Enter your question (or 'quit' to exit): ")

        # Check for quit command
        if user_query.lower() == 'quit':
            print("Exiting program...")
            break

        # Process query and print response
        try:
            response = agent_manager.query(user_query)
            print("\nResponse:")
            print(response)
            print("\n" + "-"*50 + "\n")  # Separator for readability
        except Exception as e:
            print(f"Error processing query: {str(e)}")