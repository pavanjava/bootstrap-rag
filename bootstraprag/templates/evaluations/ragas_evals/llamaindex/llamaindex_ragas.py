import os
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank, LongContextReorder
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from datasets import Dataset
import qdrant_client
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
    context_entity_recall
)
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


class LlamaIndexEvaluator:
    def __init__(self, input_dir="../data", model_name=os.environ.get("llm_model"), base_url=os.environ.get("llm_url")):
        logger.info(
            f"Initializing LlamaIndexEvaluator with input_dir={input_dir}, model_name={model_name}, base_url={base_url}")

        Settings.llm = Ollama(model=os.environ.get('llm_model'), base_url=base_url)
        Settings.embed_model = OllamaEmbedding(model_name=os.environ.get('embed_model_name'), base_url=base_url)

        self.input_dir = input_dir
        self.model_name = model_name
        self.base_url = base_url
        self.documents = self.load_documents()
        self.index_data(documents=self.documents)
        self.top_k = int(os.environ.get('retrieve_top_k'))

        logger.info(f"Loading dataset from {os.environ.get('gold_dataset_file')}")
        self.dataset = pd.read_json(os.path.join(input_dir, os.environ.get('gold_dataset_file')))
        self.query_engine = self.build_query_engine(postprocessing_method=os.environ.get('postprocessing_method'))

    def load_documents(self):
        logger.info(f"Loading documents from directory: {self.input_dir}")
        documents = SimpleDirectoryReader(input_dir=self.input_dir, required_exts=['.pdf'],
                                          num_files_limit=30).load_data(
            show_progress=True)
        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def index_data(self, documents):
        logger.info("Indexing data in Qdrant")
        # Initialize Qdrant client
        qdrant_cli = qdrant_client.QdrantClient(url=os.environ.get("qdrant_url"),
                                                api_key=os.environ.get("qdrant_api_key"))
        qdrant_vector_store = QdrantVectorStore(collection_name=os.environ.get("collection_name"), client=qdrant_cli)
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
        if not qdrant_cli.collection_exists(collection_name=os.environ.get("collection_name")):
            VectorStoreIndex.from_documents(storage_context=storage_context, documents=documents, show_progress=True)
        logger.info("Indexing data in Qdrant finished")

    def build_query_engine(self, postprocessing_method: str):
        logger.info("Initializing Qdrant client")
        # Initialize Qdrant client
        qdrant_cli = qdrant_client.QdrantClient(url=os.environ.get("qdrant_url"),
                                                api_key=os.environ.get("qdrant_api_key"))
        logger.info("Setting up Qdrant vector store")
        # Set up Qdrant vector store as per your experiment hybrid vs normal
        qdrant_vector_store = QdrantVectorStore(collection_name=os.environ.get("collection_name"), client=qdrant_cli)
        # qdrant_vector_store = QdrantVectorStore(collection_name=os.environ.get("collection_name"),
        #                                         client=qdrant_cli,
        #                                         fastembed_sparse_model=os.environ.get("sparse_model"),
        #                                         enable_hybrid=True
        #                                         )

        logger.info("Building VectorStoreIndex from existing collection")
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store,
            embed_model=OllamaEmbedding(model_name=os.environ.get("embed_model_name"), base_url=self.base_url)
        )
        query_engine = None
        if os.environ.get('enable_postprocessing_method') == 'true' and postprocessing_method == 'llm_reranker':
            reranker = LLMRerank(llm=Settings.llm, choice_batch_size=self.top_k)
            query_engine = vector_index.as_query_engine(top_k=self.top_k, node_postprocessors=[reranker])
        elif os.environ.get(
                'enable_postprocessing_method') == 'true' and postprocessing_method == 'sentence_transformer_rerank':
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=self.top_k
            )
            query_engine = vector_index.as_query_engine(top_k=self.top_k, node_postprocessors=[reranker])
        elif os.environ.get(
                'enable_postprocessing_method') == 'true' and postprocessing_method == 'long_context_reorder':
            reorder = LongContextReorder()
            query_engine = vector_index.as_query_engine(top_k=self.top_k, node_postprocessors=[reorder])
        else:
            query_engine = vector_index.as_query_engine(similarity_top_k=self.top_k, llm=Settings.llm)
        logger.info(f"Query engine built successfully: {query_engine}")
        return query_engine

    def generate_responses(self, test_questions, test_answers=None):

        logger.info("Generating responses for test questions")
        responses = [self.query_engine.query(q) for q in test_questions]

        answers = []
        contexts = []
        for r in responses:
            answers.append(r.response)
            contexts.append([c.node.get_content() for c in r.source_nodes])

        dataset_dict = {
            "question": test_questions,
            "answer": answers,
            "contexts": contexts,
        }
        if test_answers is not None:
            dataset_dict["ground_truth"] = test_answers
        logger.info("Responses generated successfully")
        return Dataset.from_dict(dataset_dict)

    def evaluate(self):
        logger.info("Starting evaluation")
        evaluation_ds = self.generate_responses(
            test_questions=self.dataset["question"].tolist(),
            test_answers=self.dataset["ground_truth"].tolist()
        )
        metrics = [
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
            context_entity_recall
        ]
        logger.info("Evaluating dataset")
        evaluation_result = evaluate(
            dataset=evaluation_ds,
            metrics=metrics
        )
        logger.info("Evaluation completed")
        return evaluation_result


if __name__ == "__main__":
    evaluator = LlamaIndexEvaluator()
    _evaluation_result = evaluator.evaluate()
    logger.info("Evaluation report generated:")
    _evaluation_result.to_pandas().to_csv('evaluation_report.csv')
