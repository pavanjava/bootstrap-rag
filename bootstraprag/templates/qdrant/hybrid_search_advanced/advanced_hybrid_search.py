import os
import tqdm
from qdrant_client import QdrantClient, models
from fastembed.embedding import TextEmbedding
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset

_ = load_dotenv(find_dotenv())


class AdvancedHybridSearch:
    def __init__(self, collection_name: str):
        self.dense_embedding_model = TextEmbedding(model_name=os.environ.get("DENSE_MODEL"))
        self.sparse_embedding_model = SparseTextEmbedding(model_name=os.environ.get("SPARSE_MODEL"))
        self.late_interaction_embedding_model = LateInteractionTextEmbedding(os.environ.get("LATE_INTERACTION_MODEL"))

        self.client = QdrantClient(url=os.environ['DB_URL'], api_key=os.environ['DB_API_KEY'])

        self.collection_name = collection_name
        self.dense_embeddings = None
        self.sparse_embeddings = None
        self.late_interaction_embeddings = None
        self.dataset = None

        self._create_collection()

    def _get_dimensions(self):
        self.dataset = load_dataset("BeIR/scifact", 'corpus', split="corpus")
        self.dense_embeddings = list(self.dense_embedding_model.passage_embed(self.dataset["text"][0:1]))
        self.sparse_embeddings = list(self.sparse_embedding_model.passage_embed(self.dataset["text"][0:1]))
        self.late_interaction_embeddings = list(
            self.late_interaction_embedding_model.passage_embed(self.dataset["text"][0:1]))

    def _create_collection(self):

        self._get_dimensions()

        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "all-MiniLM-L6-v2": models.VectorParams(
                        size=len(self.dense_embeddings[0]),
                        distance=models.Distance.COSINE
                    ),
                    "colbertv2.0": models.VectorParams(
                        size=len(self.late_interaction_embeddings[0][0]),
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                },
                sparse_vectors_config={
                    "splade-PP-en-v1": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )

    def insert_data(self):
        batch_size = 4
        for batch in tqdm.tqdm(self.dataset.iter(batch_size=batch_size), total=len(self.dataset) // batch_size):
            dense_embeddings = list(self.dense_embedding_model.passage_embed(batch["text"]))
            sparse_embeddings = list(self.sparse_embedding_model.passage_embed(batch["text"]))
            late_interaction_embeddings = list(self.late_interaction_embedding_model.passage_embed(batch["text"]))

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=int(batch["_id"][i]),
                        vector={
                            "all-MiniLM-L6-v2": dense_embeddings[i].tolist(),
                            "splade-PP-en-v1": sparse_embeddings[i].as_object(),
                            "colbertv2.0": late_interaction_embeddings[i].tolist(),
                        },
                        payload={
                            "_id": batch["_id"][i],
                            "title": batch["title"][i],
                            "text": batch["text"][i],
                        }
                    )
                    for i, _ in enumerate(batch["_id"])
                ]
            )

    def query_with_dense_embedding(self, query_text: str):
        query_vector = next(self.dense_embedding_model.embed(query_text)).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="all-MiniLM-L6-v2",
            with_payload=False,
            limit=10,
        )
        return results

    def query_with_sparse_embedding(self, query_text: str):
        query_vector = next(self.sparse_embedding_model.embed(query_text))
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(**query_vector.as_object()),
            using="splade-PP-en-v1",
            with_payload=False,
            limit=10,
        )
        return results

    def query_with_late_interaction_embedding(self, query_text: str):
        query_vector = next(self.late_interaction_embedding_model.embed(query_text)).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="colbertv2.0",
            with_payload=False,
            limit=10,
        )
        return results

    def query_with_rrf(self, query_text: str):
        dense_query_vector = next(self.dense_embedding_model.embed(query_text)).tolist()
        sparse_query_vector = next(self.sparse_embedding_model.embed(query_text))

        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using="all-MiniLM-L6-v2",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_query_vector.as_object()),
                using="splade-PP-en-v1",
                limit=20,
            ),
        ]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            with_payload=False,
            limit=10,
        )
        return results