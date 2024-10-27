import os

from llm_as_judge import LLMasJudge
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm_as_judge = LLMasJudge(
    file_path='data/mlops.pdf',
    collection_name=os.environ.get("COLLECTION_NAME"),
    qdrant_url=os.environ.get("QDRANT_DB_URL"),
    qdrant_api_key=os.environ.get("QDRANT_DB_KEY")
)

q = "what are challenges of mlops?"
llm_as_judge.retrieval_grader(question=q)
ans = llm_as_judge.generate(question=q)
print(ans)
llm_as_judge.hallucination_grader(question=q, generation=ans)
llm_as_judge.answer_grader(question=q, generation=ans)
