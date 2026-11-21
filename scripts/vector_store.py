"""Qdrant adapter for storing and searching research passages."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models


COLLECTION_NAME = "ai_text_detection_research"
QDRANT_URL = "http://127.0.0.1:6333"


class VectorStore:
    def __init__(self, url: str = QDRANT_URL, collection: str = COLLECTION_NAME) -> None:
        self.client = QdrantClient(url=url, prefer_grpc=False)
        self.collection = collection

    def recreate_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert(self, documents: list[dict], vectors, batch_size: int = 128) -> None:
        for start in range(0, len(documents), batch_size):
            points = []
            for document, vector in zip(
                documents[start : start + batch_size], vectors[start : start + batch_size]
            ):
                stable_id = str(uuid.uuid5(uuid.NAMESPACE_URL, document["id"]))
                points.append(
                    models.PointStruct(
                        id=stable_id,
                        vector=vector.tolist(),
                        payload=document,
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query_vector, limit: int = 5) -> list:
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )
        return result.points
