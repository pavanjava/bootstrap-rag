from abc import ABC
from dotenv import load_dotenv, find_dotenv
from react_agent_with_query_engine import ReActWithQueryEngine
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class ReactRAGServingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.react_with_engine = None

    def setup(self, devices):
        self.react_with_engine = ReActWithQueryEngine(input_dir='data', show_progress=True)

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, query: str):
        try:
            return self.react_with_engine.query(user_query=query)
        except Exception as e:
            return e.args[0]

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = ReactRAGServingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
