from abc import ABC
from rag_with_citation import CitationQueryEngineRAG
from dotenv import load_dotenv, find_dotenv
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class CitationRAGAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.citation_rag: CitationQueryEngineRAG = None

    def setup(self, devices):
        self.citation_rag = CitationQueryEngineRAG()

    def decode_request(self, request, **kwargs):
        return request["query"]

    def predict(self, query: str):
        try:
            return self.citation_rag.query(question=query)
        except Exception as e:
            return e.args[0]

    def encode_response(self, output, **kwargs):
        return {'assistant': output}


if __name__ == '__main__':
    api = CitationRAGAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat/completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
