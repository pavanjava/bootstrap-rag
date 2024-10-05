import os

from fastembed import TextEmbedding, ImageEmbedding
from qdrant_client import QdrantClient, models
from PIL import Image
from typing import List
from dotenv import load_dotenv, find_dotenv


class MultiModalSearch:
    def __init__(self, documents: List[dict]):
        _ = load_dotenv(find_dotenv())
        self.documents = documents
        self.collection_name = os.environ.get('COLLECTION_NAME')
        text_model_name = os.environ.get('TEXT_MODEL')  # CLIP text encoder
        self.text_model = TextEmbedding(model_name=text_model_name)
        self.text_embeddings_size = self.text_model._get_model_description(text_model_name)[
            "dim"]  # dimension of text embeddings, produced by CLIP text encoder (512)
        self.texts_embeded = list(
            self.text_model.embed(
                [document["caption"] for document in documents]))  # embedding captions with CLIP text encoder

        image_model_name = os.environ.get('IMAGE_MODEL')  # CLIP image encoder
        self.image_model = ImageEmbedding(model_name=image_model_name)
        self.image_embeddings_size = self.image_model._get_model_description(image_model_name)[
            "dim"]  # dimension of image embeddings, produced by CLIP image encoder (512)
        self.images_embeded = list(
            self.image_model.embed(
                [document["image"] for document in documents]))  # embedding images with CLIP image encoder

        self.client = QdrantClient(url=os.environ.get('DB_URL'), api_key=os.environ.get('DB_API_KEY'))

    # this method will create the collection if dones not exist and inserts  the data into it
    def _create_and_insert(self):
        if not self.client.collection_exists(self.collection_name):  # creating a Collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={  # Named Vectors
                    "image": models.VectorParams(size=self.image_embeddings_size, distance=models.Distance.COSINE),
                    "text": models.VectorParams(size=self.text_embeddings_size, distance=models.Distance.COSINE),
                }
            )

        self.client.upload_points(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=idx,  # unique id of a point, pre-defined by the user
                    vector={
                        "text": self.texts_embeded[idx],  # embeded caption
                        "image": self.images_embeded[idx]  # embeded image
                    },
                    payload=doc  # original image and its caption
                )
                for idx, doc in enumerate(self.documents)
            ]
        )

    def search_image_by_text(self, user_query: str):
        find_image = self.text_model.embed(
            [
                "suggest an architecture for designing Vision RAG platform"])  # query, we embed it, so it also becomes a vector

        image_path = self.client.search(
            collection_name=self.collection_name,  # searching in our collection
            query_vector=("image", list(find_image)[0]),  # searching only among image vectors with our textual query
            with_payload=["image"],
            # user-readable information about search results, we are interested to see which image we will find
            limit=1  # top-1 similar to the query result
        )[0].payload['image']

        Image.open(image_path).show()

    def search_text_by_image(self, image_path: str):
        find_image = self.image_model.embed([image_path])  # embedding our image query

        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=("text", list(find_image)[0]),
            # now we are searching only among text vectors with our image query
            with_payload=["caption"],
            # user-readable information about search results, we are interested to see which caption we will get
            limit=1
        )[0].payload['caption']

        return response
