# Processed research corpus

`chunks.jsonl` contains one JSON object per retrieval chunk. Each object includes:

- a stable chunk ID and source filename;
- publication title, year, type, and canonical URL from `SOURCES.md`;
- the detected section and inclusive PDF page range;
- the extracted text and its word count.

The builder extracts text page by page, groups paragraphs under detected section headings, targets about 600 words per chunk, caps chunks at 800 words, and carries up to 100 words into the next chunk for context. Long paragraphs are split at sentence boundaries when possible. Page numbers refer to the PDF pages.

Rebuild from the project directory with:

```sh
python3 rag_research_assistant/scripts/build_chunks.py
```

PDF text extraction can flatten tables and may leave some page headers or footers in the text. Review those cases before using the corpus for final answers, especially when a question depends on a table value.
