#!/usr/bin/env python3
"""Show source passages found by the local semantic search."""

from __future__ import annotations

import argparse

from retrieval import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question in Russian or English")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    documents = retrieve(args.question, top_k=args.top_k)
    if not documents:
        raise SystemExit("No matching passages found. Check that Qdrant contains the corpus.")
    print("Подходящие фрагменты из корпуса:")
    for rank, document in enumerate(documents, start=1):
        pages = (
            str(document["page_start"])
            if document["page_start"] == document["page_end"]
            else f"{document['page_start']}–{document['page_end']}"
        )
        print(f"\n{rank}. {document['title']} ({document['year']})")
        print(f"   Цитирование: [{document['title']}, {document['year']}, PDF с. {pages}]")
        print(f"   Раздел: {document['section']}")
        print(f"   Сходство: {document['similarity']:.3f} (ранжирование, не вероятность)")
        print(f"   Ссылка: {document['source_url']}")
        print(f"   {document['text'][:700].strip()}" + ("…" if len(document['text']) > 700 else ""))


if __name__ == "__main__":
    main()
