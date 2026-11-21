"""Retrieve passages from the local Qdrant research collection."""

from __future__ import annotations

from embedding_client import EmbeddingClient
from vector_store import VectorStore


def _matches_explicit_condition(question: str, document: dict) -> bool:
    """Avoid treating generic robustness passages as evidence about paraphrasing."""
    normalized_question = question.casefold()
    if "paraphras" in normalized_question or "перефраз" in normalized_question:
        text = document.get("text", "").casefold()
        return "paraphras" in text or "перефраз" in text
    return True


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    embedder = EmbeddingClient()
    query_vector = embedder.encode_query(question)
    store = VectorStore()
    candidates = store.search(query_vector, limit=max(top_k * 20, top_k))

    selected: list[dict] = []
    seen_ranges: list[tuple[str, str, int, int]] = []
    for point in candidates:
        document = dict(point.payload or {})
        if not document:
            continue
        if not _matches_explicit_condition(question, document):
            continue
        current = (
            document["filename"],
            document["section"],
            int(document["page_start"]),
            int(document["page_end"]),
        )
        overlaps_existing = any(
            prior[0] == current[0]
            and prior[1] == current[1]
            and current[2] <= prior[3]
            and prior[2] <= current[3]
            for prior in seen_ranges
        )
        if overlaps_existing:
            continue
        seen_ranges.append(current)
        document["similarity"] = float(point.score)
        selected.append(document)
        if len(selected) >= max(1, top_k):
            break
    return selected
