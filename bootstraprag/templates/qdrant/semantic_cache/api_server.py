from abc import ABC
from semantic_cache import SemanticCache, compute_response
import litserve as ls
from dotenv import load_dotenv, find_dotenv
import os


class SemanticCacheAPI(ls.LitAPI, ABC):
    def __init__(self):
        load_dotenv(find_dotenv())
        self.semantic_cache: SemanticCache = None

    def setup(self, device):
        self.semantic_cache = SemanticCache()

    def decode_request(self, request, **kwargs):
        return request['question']

    def predict(self, query, **kwargs):
        return self.semantic_cache.get_response(query=query, compute_response_func=compute_response)

    def encode_response(self, output, **kwargs):
        return {"response": output}


if __name__ == '__main__':
    api = SemanticCacheAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))