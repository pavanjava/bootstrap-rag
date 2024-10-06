import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.qdrant import QdrantVectorStore
from rag_evaluator import RAGEvaluator
from dotenv import load_dotenv, find_dotenv
import qdrant_client
import logging


class ControllableAgentsWithHumanInLoop:
    def __init__(self, input_dir: str, show_progress: bool = True, required_exts: list[str] = ['.pdf', '.txt'],
                 similarity_top_k: int = 3, chunk_size: int = 512, chunk_overlap: int = 200, max_iterations: int = 20):
        load_dotenv(find_dotenv())

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        llm = OpenAI(model=os.environ.get('OPENAI_QUERY_MODEL'))

        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        Settings.llm = llm

        self.rag_evaluator = RAGEvaluator()

        self.similarity_top_k = similarity_top_k

        self.text_parser = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)

        self.client = qdrant_client.QdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=os.environ['COLLECTION_NAME'])
        self.vector_index = None

        self.mlops_data = SimpleDirectoryReader(input_dir=input_dir, required_exts=required_exts).load_data(
            show_progress=show_progress)

        self.mlops_tool = self.get_tool("mlops_tool", "MLOps Tool", documents=self.mlops_data)

        self.query_engine_tools = [self.mlops_tool]

        agent_llm = OpenAI(model=os.environ.get('OPENAI_AGENT_MODEL'))
        self.agent = ReActAgent.from_tools(
            self.query_engine_tools, llm=agent_llm, verbose=True, max_iterations=max_iterations
        )

    def get_tool(self, name, full_name, documents=None):
        self.logger.info("initializing the storage context")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        if not self.client.collection_exists(collection_name=os.environ.get('COLLECTION_NAME')):
            self.logger.info("indexing the nodes in VectorStoreIndex")
            self.vector_index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                transformations=Settings.transformations,
            )
        else:
            self.vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        query_engine = self.vector_index.as_query_engine(similarity_top_k=self.similarity_top_k, llm=Settings.llm)
        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=name,
                description=(
                    "Provides information about mlops and its details"
                    f" {full_name}"
                ),
            ),
        )
        return query_engine_tool

    def chat_repl(self, user_query: str, exit_when_done: bool = True):
        task_message = user_query

        task = self.agent.create_task(task_message)

        response = None
        step_output = None
        message = None
        while message != "exit":
            if message is None or message == "":
                step_output = self.agent.run_step(task.task_id)
            else:
                step_output = self.agent.run_step(task.task_id, input=message)
            if exit_when_done and step_output.is_last:
                print(">> Task marked as finished by the agent, executing task execution.")
                break

            message = input(">> Add feedback during step? (press enter/leave blank to continue, "
                            "and type 'exit' to stop): ")
            if message == "exit":
                break

        if step_output is None:
            print(">> You haven't run the agent. Task is discarded.")
        elif not step_output.is_last:
            print(">> The agent hasn't finished yet. Task is discarded.")
        else:
            response = self.agent.finalize_response(task.task_id)
        print(f"Agent: {str(response)}")

        if os.environ.get('IS_EVALUATION_NEEDED') == 'true':
            self.rag_evaluator.evaluate(user_query=user_query, response_obj=response)
        return response
