# driver code
from self_correction_core import SelfCorrectingRAG


self_correcting_rag = SelfCorrectingRAG(input_dir='data', show_progress=True, no_of_retries=3)

# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query [type 'bye' to 'exit']: ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye" or user_query.lower() == "exit":
        break

    response1 = self_correcting_rag.query_with_retry_query_engine(query=user_query)
    print(response1)

    response1 = self_correcting_rag.query_with_retry_query_engine(query=user_query)
    print(response1)

    response1 = self_correcting_rag.query_with_retry_query_engine(query=user_query)
    print(response1)