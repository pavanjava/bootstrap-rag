from dotenv import load_dotenv, find_dotenv
from recursive_retriever_agents_core import RecursiveAgentManager
import litserve as lit
import os


class RecursiveAgentsAPI(lit.LitAPI):
    def __init__(self):
        load_dotenv(find_dotenv())
        self.agent_names = ['mlops', 'attention', 'orthodontics']
        self.agent_manager = None

    def setup(self, device):
        self.agent_manager = RecursiveAgentManager(self.agent_names)

    def decode_request(self, request, **kwargs):
        return request['query']

    def predict(self, x, **kwargs):
        return self.agent_manager.query(x)

    def encode_response(self, output, **kwargs):
        return {'Agent': output}


if __name__ == "__main__":
    lit_api = RecursiveAgentsAPI()
    server = lit.LitServer(lit_api=lit_api, api_path='/api/v1/chat-completion',
                           workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
