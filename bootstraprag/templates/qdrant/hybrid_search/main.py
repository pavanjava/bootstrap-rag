import os

from search import HybridSearch
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

simple_search = HybridSearch(collection_name=os.environ['COLLECTION_NAME'])

# uncomment if you want to index the data for the first time
# simple_search.insert()
result = simple_search.search(input_text='San Francisco')

print(result)