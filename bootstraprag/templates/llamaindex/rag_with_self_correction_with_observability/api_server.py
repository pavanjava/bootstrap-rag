from abc import ABC
from dotenv import load_dotenv, find_dotenv
from self_correction_core import SelfCorrectingRAG
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class SimpleRAGServingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.self_correcting_rag = None
        self.method: str | None = None

    def setup(self, devices):
        self.method = 'retry_query_engine'  # default query engine
        self.self_correcting_rag = SelfCorrectingRAG(input_dir='data', show_progress=True, no_of_retries=3)

    def decode_request(self, request, **kwargs):
        if "method" in request.keys():
            self.method = request["method"]
        return request["query"]

    def predict(self, query: str):
        try:
            if self.method == 'retry_query_engine':
                return self.self_correcting_rag.query_with_retry_query_engine(query=query)
            elif self.method == 'source_query_engine':
                return self.self_correcting_rag.query_with_source_query_engine(query=query)
            elif self.method == 'guideline_query_engine':
                return self.self_correcting_rag.query_with_guideline_query_engine(query=query)
            else:
                raise Exception('No a proper method passed')
        except Exception as e:
            return e.args[0]

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = SimpleRAGServingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
