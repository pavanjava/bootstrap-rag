from abc import ABC
from dotenv import load_dotenv, find_dotenv
from llm_as_judge import LLMasJudge
import litserve as ls
import os

_ = load_dotenv(find_dotenv())


class LLMasJudgeAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.llm_as_judge: LLMasJudge = None
        self.FILE_PATH = 'data/mlops.pdf'
        self.COLLECTION_NAME = os.environ.get('COLLECTION_NAME')
        self.QDRANT_URL = os.environ.get('QDRANT_DB_URL')
        self.QDRANT_API_KEY = os.environ.get('QDRANT_DB_KEY')
        self.operation_name: str = ''

    def setup(self, devices):
        self.llm_as_judge = LLMasJudge(
            file_path=self.FILE_PATH,
            collection_name=self.COLLECTION_NAME,
            qdrant_url=self.QDRANT_URL,
            qdrant_api_key=self.QDRANT_API_KEY
        )

    def decode_request(self, request, **kwargs):
        self.operation_name = request["operation"]
        return request["query"]

    def predict(self, query: str):
        if self.operation_name == 'retrieval_grader':
            return self.llm_as_judge.retrieval_grader(question=query)
        elif self.operation_name == 'generate':
            return self.llm_as_judge.generate(question=query)
        elif self.operation_name == 'hallucination_grader':
            generation = self.llm_as_judge.generate(question=query)
            return self.llm_as_judge.hallucination_grader(question=query, generation=generation)
        elif self.operation_name == 'answer_grader':
            generation = self.llm_as_judge.generate(question=query)
            return self.llm_as_judge.answer_grader(question=query, generation=generation)

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = LLMasJudgeAPI()
    server = ls.LitServer(lit_api=api, api_path='/v1/chat/completions',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
