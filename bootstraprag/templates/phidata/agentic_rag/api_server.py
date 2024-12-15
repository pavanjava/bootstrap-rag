from abc import ABC
from dotenv import load_dotenv, find_dotenv
from agentic_rag import qdrant_agent
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class ReactRAGServingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.ai_agent = None
        self.user_id = "pavan_mantha"

    def setup(self, devices):
        self.ai_agent = qdrant_agent(user='pavan_mantha')

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, query: str, **kwargs):
        try:
            response = self.ai_agent.run(message=query)
            # you can get the memory value also.
            # print(self.ai_tutor.get_memories(user_id=self.user_id))
            return response
        except Exception as e:
            return e.args[0]

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = ReactRAGServingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
