import os

from simple_rag import SimpleRAG
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

simpleRag = SimpleRAG(
    file_path='data/mlops.pdf',
    collection_name=os.environ.get("COLLECTION_NAME"),
    qdrant_url=os.environ.get("QDRANT_DB_URL"),
    qdrant_api_key=os.environ.get("QDRANT_DB_KEY")
)

'''Uncomment the following line to insert data (only needed once) explicitly,
else the data is inserted on the initialization'''
# simpleRag.insert_data_with_metadata()

# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query [type 'bye' to 'exit']: ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye" or user_query.lower() == "exit":
        break

    response = simpleRag.query(user_query=user_query)
    print(f"Answer: {response}")