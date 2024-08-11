from base_rag import BaseRAG

# this step will do pre processing, indexing in vector store, creating retriever (hyDE).
# this may take some time based on your document size and chunk strategy.
base_rag = BaseRAG(show_progress=True,
                   data_path='data')  # leaving all the defaults. if needed override them in constructor
# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query [type 'bye' to 'exit']: ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye" or user_query.lower() == "exit":
        break

    response = base_rag.query(query_string=user_query)
    print(response)
