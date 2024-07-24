import os
from uuid import uuid4
from qdrant_client import qdrant_client, models
from dotenv import find_dotenv, load_dotenv
from qdrant_client.http.models import Distance
from qdrant_client.conversions.common_types import QueryResponse, UpdateResult


class SimpleSearch:
    _ = load_dotenv(find_dotenv())

    def __init__(self, collection_name: str, vector_dimension: int = 5, distance: Distance = models.Distance.COSINE):
        self.client = qdrant_client.QdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])
        self.vector_dimension = vector_dimension
        self.distance = distance
        self.collection_name = collection_name
        self._create_collection(collection_name=collection_name)

    def _create_collection(self, collection_name):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.vector_dimension, distance=self.distance))

    def insert(self) -> UpdateResult:
        # simple boilerplate code adjust it accordingly
        response: UpdateResult = self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid4()),
                    payload={
                        "text": "this is auto generated bootstrap code",
                    },
                    vector=[0.9, 0.1, 0.1, 0.4, -0.8],
                ),
            ],
        )

        return response

    def search(self) -> QueryResponse:
        # simple boilerplate code adjust it accordingly
        response: QueryResponse = self.client.query_points(
            collection_name=self.collection_name,
            query=[0.9, 0.1, 0.2, 0.4, -0.6], # <--- Dense vector
        )

        return response
