from dotenv import load_dotenv, find_dotenv
from sub_question_query_engine import SubQuestionQueryEngineAgent
import litserve as lit
import os


class SubQuestionQueryAPI(lit.LitAPI):
    def __init__(self):
        load_dotenv(find_dotenv())
        self.engine = None

    def setup(self, device):
        self.engine = SubQuestionQueryEngineAgent()

    def decode_request(self, request, **kwargs):
        return request['query']

    def predict(self, x, **kwargs):
        return self.engine.query(x)

    def encode_response(self, output, **kwargs):
        return {'Agent': output}


if __name__ == "__main__":
    lit_api = SubQuestionQueryAPI()
    server = lit.LitServer(lit_api=lit_api, api_path='/api/v1/chat-completion',
                           workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
