from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import LlamaIndexTool
from llama_index.core.tools import FunctionTool
from llama_index_query_engine import RagQueryEngine
from typing import Any, Dict
import nltk

nltk.download('punkt')

rag_query_engine: RagQueryEngine = RagQueryEngine(input_dir='data', show_progress=True)


def get_valid_property(data) -> str:
    # Return the first non-empty, non-"None" value among the keys
    return (
        data["description"] if data["description"] not in [None, "None"] else None
                                                                              or data["text"] if data["text"] not in [
            None, "None"] else None or
                               data["content"] if data["content"] not in [None, "None"] else None
    )


def use_query_engine(query: Dict):
    """Use this function to get answers for mlops

    Args:
        query (Dict): the user query to search.
    """
    query_engine = rag_query_engine.get_query_engine()
    user_query = get_valid_property(
        data=query
    )
    return query_engine.query(str_or_query_bundle=user_query)


query_engine_tool = FunctionTool.from_defaults(
    use_query_engine,
    name="mlops tool",
    description="Use this tool to lookup questions regarding mlops"
)
tool = LlamaIndexTool.from_tool(query_engine_tool)

llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)

# Initialize Tool from a LlamaIndex Query Engine
# query_engine = rag_query_engine.get_query_engine()
# query_tool = LlamaIndexTool.from_query_engine(
#     query_engine,
#     name="mlops tool",
#     description="Use this tool to lookup questions regarding mlops"
# )

# Create and assign the tools to an agent
rag_agent = Agent(
    llm=llm,
    role='Senior MLops export',
    goal='Provide up-to-date answer on the user query regarding {topic}',
    backstory="""As an mlops expert use the tool for fetching the proper context for answering 
              the user query regarding {topic}""",
    tools=[tool],
    max_iter=10,
    memory=True
)

rag_task = Task(
    description="{topic}",
    expected_output="A summarizing answer for question {topic}.",
    agent=rag_agent
)

rag_crew = Crew(
    agents=[rag_agent],
    tasks=[rag_task],
    process=Process.sequential,
    verbose=True
)

# Example of using kickoff_async
# inputs = {'topic': 'what are the challenges of mlops?'}
# output = rag_crew.kickoff(inputs=inputs)
# print(output)
