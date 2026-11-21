#!/usr/bin/env python3
"""Build embeddings and load all research passages into local Qdrant."""

from __future__ import annotations

from build_index import build_artifacts
from vector_store import COLLECTION_NAME, VectorStore


def main() -> None:
    documents, vectors = build_artifacts()
    store = VectorStore()
    store.recreate_collection(vector_size=int(vectors.shape[1]))
    store.upsert(documents, vectors)
    print(f"Loaded {len(documents)} passages into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
