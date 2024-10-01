from llama_index.core import SimpleDirectoryReader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv, find_dotenv
import logging
import os

_ = load_dotenv(find_dotenv())
logging.basicConfig(level=int(os.environ['INFO']))
logger = logging.getLogger(__name__)


class TestSetGenerator:
    def __init__(self):
        # generator with ollama models
        # generator_llm = Ollama(model=os.environ.get('OLLAMA_LLM_MODEL'), request_timeout=300)
        # critic_llm = Ollama(model=os.environ.get('OLLAMA_LLM_MODEL'), request_timeout=300)
        # embeddings = OllamaEmbedding(model_name=os.environ.get('OLLAMA_EMBED_MODEL'), request_timeout=300)

        # generator with openai models
        generator_llm = OpenAI(model=os.environ.get('OPENAI_MODEL'), timeout=300)
        critic_llm = OpenAI(model=os.environ.get('OPENAI_MODEL'), timeout=300)
        embeddings = OpenAIEmbedding(model=os.environ.get('OPENAI_EMBED_MODEL'), timeout=300)

        self.generator = TestsetGenerator.from_llama_index(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings=embeddings,
        )

    def generate_test_set(self, input_dir, test_size: int = 5, simple_dist: float = 0.5, reasoning_dist: float = 0.25,
                          multi_context_dist: float = 0.25, show_progress: bool = True):
        logger.info('loading docs..')
        _docs = SimpleDirectoryReader(input_dir=input_dir).load_data(show_progress=show_progress)
        # generate test set
        logger.info('test set generation started..')
        test_set = self.generator.generate_with_llamaindex_docs(
            documents=_docs,
            test_size=test_size,
            distributions={simple: simple_dist, reasoning: reasoning_dist, multi_context: multi_context_dist},
        )
        logger.info('test set generation ended..filename: testset.csv')
        df = test_set.to_pandas()
        df.to_csv(filename="testset.csv")
