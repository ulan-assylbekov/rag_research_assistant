#!/usr/bin/env python3
"""Extract research PDFs into page-attributed, paragraph-aware JSONL chunks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "chunks.jsonl"
SOURCES = ROOT / "SOURCES.md"
TARGET_WORDS = 600
MAX_WORDS = 800
OVERLAP_WORDS = 100

HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[A-Z])\.?\s+)?"
    r"(?:abstract|introduction|background|related work|method(?:ology)?|"
    r"approach|experiments?|results?|discussion|limitations?|conclusion|"
    r"references|data|evaluation|benchmark|threats? to validity)\b.*$",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z].{2,100}$")


def source_metadata() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for line in SOURCES.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "`" not in line or "http" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        filename = cells[0].strip("`")
        if filename.endswith(".pdf"):
            result[filename] = {
                "title": cells[1],
                "year": cells[2],
                "document_type": cells[3],
                "source_url": cells[4],
            }
    return result


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100 or len(stripped.split()) > 7:
        return False
    if "&" in stripped or re.search(r"\b\d{2,}\b", stripped):
        return False
    return bool(HEADING_RE.match(stripped) or NUMBERED_HEADING_RE.match(stripped))


def extract_paragraphs(pdf_path: Path) -> list[dict]:
    paragraphs: list[dict] = []
    for page_number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        text = page.extract_text() or ""
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        current: list[str] = []
        for line in lines + [""]:
            if not line:
                if current:
                    paragraphs.append({"page": page_number, "text": " ".join(current)})
                    current = []
            elif is_heading(line):
                if current:
                    paragraphs.append({"page": page_number, "text": " ".join(current)})
                    current = []
                paragraphs.append({"page": page_number, "heading": line})
            else:
                current.append(line)
    return paragraphs


def split_long_paragraph(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    # Prefer sentence boundaries while keeping each piece below the word cap.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if current and count + len(sentence_words) > max_words:
            parts.append(" ".join(current))
            current, count = [], 0
        if len(sentence_words) > max_words:
            for start in range(0, len(sentence_words), max_words):
                piece = sentence_words[start : start + max_words]
                if current:
                    parts.append(" ".join(current))
                    current, count = [], 0
                parts.append(" ".join(piece))
        else:
            current.append(sentence)
            count += len(sentence_words)
    if current:
        parts.append(" ".join(current))
    return parts


def make_chunks(paragraphs: list[dict], metadata: dict, filename: str) -> list[dict]:
    chunks: list[dict] = []
    section = "Front matter"
    buffer: list[dict] = []
    buffer_words = 0

    def tail_overlap(items: list[dict]) -> list[dict]:
        remaining = OVERLAP_WORDS
        tail: list[dict] = []
        for prior in reversed(items):
            words = prior["text"].split()
            take = min(remaining, len(words))
            if take:
                tail.insert(0, {"page": prior["page"], "text": " ".join(words[-take:])})
                remaining -= take
            if remaining == 0:
                break
        return tail

    def emit(items: list[dict], chunk_section: str) -> None:
        if not items:
            return
        text = "\n\n".join(item["text"] for item in items if item.get("text"))
        if not text.strip():
            return
        pages = [item["page"] for item in items]
        chunks.append({
            "id": f"{Path(filename).stem}-{len(chunks) + 1:04d}",
            **metadata,
            "filename": filename,
            "section": chunk_section,
            "page_start": min(pages),
            "page_end": max(pages),
            "word_count": len(text.split()),
            "text": text,
        })

    for paragraph in paragraphs:
        if "heading" in paragraph:
            heading = paragraph["heading"]
            if buffer:
                emit(buffer, section)
                buffer = []
                buffer_words = 0
            section = heading
            continue
        # Leave room for the carried overlap in the next chunk.
        for piece in split_long_paragraph(paragraph["text"], MAX_WORDS - OVERLAP_WORDS):
            item = {"page": paragraph["page"], "text": piece}
            words = len(piece.split())
            if buffer and buffer_words + words > MAX_WORDS:
                emit(buffer, section)
                buffer = tail_overlap(buffer)
                buffer_words = sum(len(item["text"].split()) for item in buffer)
            buffer.append(item)
            buffer_words += words
            if buffer_words >= TARGET_WORDS:
                emit(buffer, section)
                buffer = tail_overlap(buffer)
                buffer_words = sum(len(item["text"].split()) for item in buffer)
    emit(buffer, section)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    metadata_by_file = source_metadata()
    all_chunks: list[dict] = []
    for pdf_path in sorted(args.input.glob("*.pdf")):
        metadata = metadata_by_file.get(pdf_path.name)
        if not metadata:
            print(f"Skipping {pdf_path.name}: not listed in SOURCES.md")
            continue
        paragraphs = extract_paragraphs(pdf_path)
        chunks = make_chunks(paragraphs, metadata, pdf_path.name)
        if not chunks:
            print(f"Warning: no extractable text in {pdf_path.name}")
        all_chunks.extend(chunks)
        print(f"{pdf_path.name}: {len(paragraphs)} paragraphs -> {len(chunks)} chunks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for chunk in all_chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
