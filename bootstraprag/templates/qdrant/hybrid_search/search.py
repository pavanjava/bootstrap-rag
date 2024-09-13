import os
import json
from tqdm import tqdm
from qdrant_client import qdrant_client, models
from dotenv import find_dotenv, load_dotenv
from qdrant_client.http.models import Distance
from qdrant_client.conversions.common_types import QueryResponse, UpdateResult


class HybridSearch:
    _ = load_dotenv(find_dotenv())

    def __init__(self, collection_name: str, vector_dimension: int = 384, distance: Distance = models.Distance.COSINE):
        self.client = qdrant_client.QdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])

        # set the dense and sparse embedding models
        self.client.set_model(os.environ.get('DENSE_MODEL'))
        self.client.set_sparse_model(os.environ.get('SPARSE_MODEL'))
        self.vector_dimension = vector_dimension
        self.distance = distance
        self.collection_name = collection_name

        self.documents = []
        self.metadata = []

        # create collection call
        self._create_collection(collection_name=collection_name)

    def _create_collection(self, collection_name):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.client.get_fastembed_vector_params(),
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params()
            )

    def _read_data(self):
        payload_path = "data/startups-mini.json"

        with open(payload_path) as fd:
            for line in fd:
                obj = json.loads(line)
                self.documents.append(obj.pop("description"))
                self.metadata.append(obj)

    def insert(self) -> UpdateResult:
        self._read_data()
        # simple boilerplate code adjust it accordingly
        self.client.add(
            collection_name=self.collection_name,
            documents=self.documents,
            metadata=self.metadata,
            # batch_size=128,  # a batch os 128 embeddings will be pushed in a single request
            ids=tqdm(range(len(self.documents)))
        )

    def search(self, input_text: str) -> QueryResponse:
        search_result = self.client.query(
            collection_name=self.collection_name,
            query_text=input_text,
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the closest results
        )
        # `search_result` contains found vector ids with similarity scores
        # along with the stored payload

        # Select and return metadata
        metadata = [hit.metadata for hit in search_result]
        return metadata  # if not return entire search_result
