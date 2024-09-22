from llama_index.core.workflow import Event
from llama_index. core. base. base_query_engine import BaseQueryEngine


class QueryEngineEvent(Event):
    """Result of running retrieval"""

    base_query_engine: BaseQueryEngine

