from sub_question_query_engine import SubQuestionQueryEngineAgent


def print_welcome_message():
    print("\n=== Orthodontics Query System ===")
    print("Type your question and press Enter")
    print("Type 'quit' to exit the program")
    print("================================\n")


if __name__ == "__main__":
    # Initialize the engine
    engine = SubQuestionQueryEngineAgent()

    # Load and index documents
    print("Initializing the system... Please wait...")
    engine.load_and_index_documents()

    # Display welcome message
    print_welcome_message()

    while True:
        try:
            # Get user input
            user_question = input("\nEnter your question: ").strip()

            # Check for quit command
            if user_question.lower() == 'quit':
                print("\nThank you for using the system. Goodbye!")
                break

            # Skip empty questions
            if not user_question:
                print("Please enter a valid question.")
                continue

            # Execute query and print response
            print("\nProcessing your question...\n")
            response = engine.query(user_question)
            print("\nResponse:", response)
            print("\n" + "-" * 50)  # Separator line

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different question.")
