import os

from advanced_hybrid_search import AdvancedHybridSearch
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

adv_search = AdvancedHybridSearch(collection_name=os.environ.get('COLLECTION_NAME'))
# adv_search.insert_data()

query_text = "What is the impact of COVID-19 on the environment?"
results = adv_search.query_with_dense_embedding(query_text=query_text)
print(results)

results = adv_search.query_with_sparse_embedding(query_text=query_text)
print(results)

results = adv_search.query_with_late_interaction_embedding(query_text=query_text)
print(results)

results = adv_search.query_with_rrf(query_text=query_text)
print(results)
