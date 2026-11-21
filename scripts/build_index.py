#!/usr/bin/env python3
"""Encode prepared chunks and save reusable local vector artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embedding_client import EmbeddingClient


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
DOCS_PATH = INDEX_DIR / "documents.jsonl"
VECTORS_PATH = INDEX_DIR / "vectors.npy"


def build_artifacts() -> tuple[list[dict], np.ndarray]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Chunk file not found: {CHUNKS_PATH}")
    source_chunks = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embedder = EmbeddingClient()
    documents = embedder.prepare_documents(source_chunks)
    vectors = embedder.encode_passages([document["text"] for document in documents])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors.astype("float32"))
    with DOCS_PATH.open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")
    print(f"Created {len(documents)} passage embeddings from {len(source_chunks)} chunks.")
    print(f"Saved reusable artifacts to {INDEX_DIR}")
    return documents, vectors


def main() -> None:
    build_artifacts()


if __name__ == "__main__":
    main()
