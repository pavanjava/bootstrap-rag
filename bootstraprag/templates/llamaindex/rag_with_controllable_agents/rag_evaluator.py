from deepeval.integrations.llama_index import (
    DeepEvalFaithfulnessEvaluator,
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalContextualRelevancyEvaluator
)
from dotenv import load_dotenv, find_dotenv
from typing import Any
import os
import logging

_ = load_dotenv(find_dotenv())
logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self):
        self.faithfulness_evaluator = DeepEvalFaithfulnessEvaluator()
        self.answer_relevancy_evaluator = DeepEvalAnswerRelevancyEvaluator()
        self.context_relevancy_evaluator = DeepEvalContextualRelevancyEvaluator()

    def evaluate(self, user_query: str, response_obj: Any):
        logger.info(f"calling evaluation, user_query: {user_query}, response_obj: {response_obj}")
        retrieval_context = [node.get_content() for node in response_obj.source_nodes]
        actual_output = response_obj.response
        faithfulness_evaluation_response = self.faithfulness_evaluator.evaluate(query=user_query, response=actual_output,
                                                                                contexts=retrieval_context)
        answer_relevancy_response = self.answer_relevancy_evaluator.evaluate(query=user_query, response=actual_output,
                                                                             contexts=retrieval_context)
        context_relevancy_response = self.context_relevancy_evaluator.evaluate(query=user_query, response=actual_output,
                                                                               contexts=retrieval_context)
        logger.info(f"faithfulness_evaluation_response: {faithfulness_evaluation_response.score}")
        logger.info(f"answer_relevancy_response: {answer_relevancy_response.score}")
        logger.info(f"context_relevancy_response: {context_relevancy_response.score}")
