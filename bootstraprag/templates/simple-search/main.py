import os

from search import SimpleSearch
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

simple_search = SimpleSearch(collection_name=os.environ['COLLECTION_NAME'])

# TODO: code actual implementation of these below methods.
simple_search.insert()
simple_search.search()