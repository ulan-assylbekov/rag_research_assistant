#!/usr/bin/env python3
"""Answer a research question using Qdrant retrieval and local Ollama."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_client import OllamaClient
from retrieval import retrieve


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system_prompt.md"


def format_context(documents: list[dict]) -> str:
    blocks = []
    for number, document in enumerate(documents, start=1):
        page_range = (
            str(document["page_start"])
            if document["page_start"] == document["page_end"]
            else f"{document['page_start']}-{document['page_end']}"
        )
        blocks.append(
            f"[{number}] {document['title']} ({document['year']}), "
            f"PDF pages {page_range}; section: {document['section']}\n"
            f"URL: {document['source_url']}\n"
            f"Passage:\n{document['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def format_sources(documents: list[dict]) -> list[str]:
    """Group retrieved passages by publication while keeping all page ranges."""
    sources: dict[tuple[str, str, str], set[int]] = {}
    for document in documents:
        key = (document["title"], document["year"], document["source_url"])
        pages = sources.setdefault(key, set())
        pages.update(range(int(document["page_start"]), int(document["page_end"]) + 1))

    formatted = []
    for (title, year, source_url), pages in sources.items():
        ordered_pages = sorted(pages)
        ranges: list[str] = []
        start = previous = ordered_pages[0]
        for page in ordered_pages[1:]:
            if page == previous + 1:
                previous = page
                continue
            ranges.append(str(start) if start == previous else f"{start}–{previous}")
            start = previous = page
        ranges.append(str(start) if start == previous else f"{start}–{previous}")
        formatted.append(
            f"- {title} ({year}), PDF с. {', '.join(ranges)}: {source_url}"
        )
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Research question in Russian or English")
    parser.add_argument("--top-k", type=int, default=2, help="Number of source passages")
    args = parser.parse_args()

    if not SYSTEM_PROMPT_PATH.exists():
        raise SystemExit(f"System prompt not found: {SYSTEM_PROMPT_PATH}")
    documents = retrieve(args.question, top_k=args.top_k)
    if not documents:
        raise SystemExit("No matching passages found. Check that Qdrant contains the corpus.")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = (
        f"Original user question:\n{args.question}\n\n"
        "Use the source passages below as your evidence. Do not use the similarity scores.\n\n"
        f"Retrieved source passages:\n{format_context(documents)}"
    )

    answer = OllamaClient().answer(system_prompt, user_prompt)
    print("\nОтвет по найденным публикациям:\n")
    print(answer or "Модель не вернула текстовый ответ.")
    print("\nИсточники, переданные модели:")
    for source in format_sources(documents):
        print(source)


if __name__ == "__main__":
    main()
