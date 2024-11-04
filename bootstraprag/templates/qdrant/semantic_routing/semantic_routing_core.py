import os
from semantic_router import RouteLayer
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.index import QdrantIndex
from dotenv import load_dotenv, find_dotenv

# load the data from env file
load_dotenv(find_dotenv())


class SemanticRouter:
    def __init__(self, qdrant_api_key: str = os.environ.get('qdrant_api_key'),
                 qdrant_url: str = os.environ.get('qdrant_url'),
                 index_name="semantic-router-index"):
        """
        Initialize the SemanticRouter with OpenAI API key and Qdrant configurations.

        :param qdrant_api_key: Your Qdrant API key.
        :param qdrant_url: URL of the Qdrant instance.
        :param index_name: Name of the Qdrant index to use.
        :param location: None not to consider the in memory instance.
        """
        self.encoder = FastEmbedEncoder(name=os.environ.get('encoder_model'))
        self.qdrant_index = QdrantIndex(url=qdrant_url, api_key=qdrant_api_key, index_name=index_name, location=None)
        self.route_layer = None

    def setup_routes(self, routes):
        """
        Set up the routing layer with the provided routes.

        :param routes: List of Route objects.
        """
        self.route_layer = RouteLayer(encoder=self.encoder, routes=routes, index=self.qdrant_index)

    def route_query(self, query):
        """
        Route a query to the appropriate route based on semantic similarity.

        :param query: The input query string.
        :return: Name of the routed route.
        """
        if not self.route_layer:
            raise ValueError("Routes have not been set up. Call setup_routes() first.")
        result = self.route_layer(query)
        return result.name
