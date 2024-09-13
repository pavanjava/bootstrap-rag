import os

from search import SimpleSearch
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

simple_search = SimpleSearch(collection_name=os.environ['COLLECTION_NAME'])

# open the below insert for the first time to index the data
# simple_search.insert()

result = simple_search.search(input_text="Chicago")
print(result)