from abc import ABC
from adjacent_context_rag import PrevNextPostprocessorDemo
from dotenv import load_dotenv, find_dotenv
import litserve as lit
import os


class AdjacentContextRAG(lit.LitAPI, ABC):
    def __init__(self):
        load_dotenv(find_dotenv())
        self.demo = None

    def setup(self, device):
        # Initialize the class with your OpenAI API key
        self.demo = PrevNextPostprocessorDemo(
            data_directory="data",
            chunk_size=256,
        )
        self.demo.load_documents()
        self.demo.parse_documents_to_nodes()
        self.demo.build_index()

    def decode_request(self, request, **kwargs):
        return request['query']

    def predict(self, x, **kwargs):
        return self.demo.query_with_postprocessor(query=x)

    def encode_response(self, output, **kwargs):
        return {"assistant": output}


if __name__ == '__main__':
    api = AdjacentContextRAG()
    server = lit.LitServer(api, api_path='/api/v1/chat/completion',
                           workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'), num_api_servers=1, generate_client_file=False, log_level="info")
