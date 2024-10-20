# driver code
from react_rag import ReActWithQueryEngine
from hyde_rag import RAGWithHyDeEngine

technique = 'react'  # 'react' or 'hyde'

# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query [type 'bye' to 'exit']: ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye" or user_query.lower() == "exit":
        break
    if technique == 'hyde':
        # this step will do pre processing, indexing in vector store, creating retriever (hyDE).
        # this may take some time based on your document size and chunk strategy.
        hyde_rag = RAGWithHyDeEngine(
            data_path='data')  # leaving all the defaults. if needed override them in constructor
        response = hyde_rag.query(query_string=user_query)
    else:
        # this may take some time based on your document size and chunk strategy.
        react_rag = ReActWithQueryEngine(input_dir='data', show_progress=True)
        response = react_rag.query(user_query=user_query)

    print(response)
