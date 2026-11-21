"""Client for multilingual passage and query embeddings."""

from __future__ import annotations

import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_INPUT_TOKENS = 480  # Stay below E5's 512-token model limit.


class EmbeddingClient:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)
        return len(encoded["input_ids"])

    def split_passage(self, text: str) -> list[str]:
        """Split a long passage into pieces the model can encode without truncation."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        parts: list[str] = []
        current = ""

        def flush() -> None:
            nonlocal current
            if current:
                parts.append(current.strip())
                current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if self._token_count("passage: " + candidate) <= MAX_INPUT_TOKENS:
                current = candidate
                continue
            flush()
            if self._token_count("passage: " + sentence) <= MAX_INPUT_TOKENS:
                current = sentence
                continue
            word_group: list[str] = []
            for word in sentence.split():
                candidate_group = " ".join(word_group + [word])
                if word_group and self._token_count("passage: " + candidate_group) > MAX_INPUT_TOKENS:
                    parts.append(" ".join(word_group))
                    word_group = [word]
                else:
                    word_group.append(word)
            if word_group:
                current = " ".join(word_group)
        flush()
        return parts or [text.strip()]

    def prepare_documents(self, source_chunks: list[dict]) -> list[dict]:
        documents: list[dict] = []
        for source in source_chunks:
            # Bibliography entries can match a query by title without being evidence
            # for the paper's own findings, so keep them out of the retrieval index.
            if re.match(r"^(references|bibliography)\b", source.get("section", ""), re.IGNORECASE):
                continue
            pieces = self.split_passage(source["text"])
            for part_number, piece in enumerate(pieces, start=1):
                document = dict(source)
                document["text"] = piece
                document["word_count"] = len(piece.split())
                if len(pieces) > 1:
                    document["id"] = f"{source['id']}-p{part_number}"
                document["embedding_model"] = self.model_name
                documents.append(document)
        return documents

    def encode_passages(self, texts: list[str], show_progress: bool = True):
        passages = ["passage: " + text for text in texts]
        return self.model.encode(
            passages,
            batch_size=16,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, text: str):
        return self.model.encode(
            "query: " + text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
