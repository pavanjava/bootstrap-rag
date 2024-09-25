from abc import ABC
from dotenv import load_dotenv, find_dotenv
from simple_rag import SimpleRAG
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class SimpleRAGServingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.simpleRAG = None

    def setup(self, devices):
        self.simpleRAG = SimpleRAG(input_dir='data', show_progress=True)

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, query: str):
        return self.simpleRAG.do_rag(user_query=query)

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = SimpleRAGServingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
