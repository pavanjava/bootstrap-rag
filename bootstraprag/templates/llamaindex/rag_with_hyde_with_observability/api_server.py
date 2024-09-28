from abc import ABC
from dotenv import load_dotenv, find_dotenv
from base_rag import BaseRAG
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class ReactRAGServingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.base_rag = None

    def setup(self, devices):
        self.base_rag = BaseRAG(show_progress=True, data_path='data')

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, query: str):
        try:
            return self.base_rag.query(query_string=query)
        except Exception as e:
            return e.args[0]

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = ReactRAGServingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
