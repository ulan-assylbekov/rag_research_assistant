# RAG Research Assistant

A local CLI tool for asking research questions about AI-generated-text detection. It searches a curated set of papers, gives the most relevant passages to a local language model, and displays an answer with paper and PDF-page citations. It is a literature assistant, not a classifier for user-submitted text.

See [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) for the idea, architecture, dataset, requirements, and limitations.
See [docs/VIDEO_DEMO_SCRIPT.md](docs/VIDEO_DEMO_SCRIPT.md) for a draft narration and demonstration sequence.

## Components

- **Dataset:** 10 research PDFs, citation manifest, and page-attributed passages.
- **Embeddings:** `intfloat/multilingual-e5-small`, shared by Russian questions and English source passages.
- **Vector database:** Qdrant running locally in Docker; it stores the passage text, citation metadata, and vectors.
- **Answer model:** Qwen3 4B Instruct running locally through Ollama.
- **Interface:** command line.

## Run the project

Start Docker Desktop and make sure Ollama is running. If it is not, open a second terminal in this folder and start the local Ollama server:

```sh
ollama serve
```

Then, from the project folder:

```sh
docker compose up -d
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct
.venv/bin/python scripts/ingest_database.py
.venv/bin/python scripts/ask.py "Насколько детекторы ИИ-текста устойчивы к перефразированию?"
```

The first run downloads the embedding and chat models. The embedding model and Qwen model are cached locally. Qdrant's persistent files are under `data/qdrant_storage/` and are excluded from Git; rerun `scripts/ingest_database.py` to recreate the collection. Bibliography sections are retained in the raw chunk file but omitted from the search index so cited-paper titles are not mistaken for the source paper's findings.

To inspect retrieved evidence without asking Qwen to synthesize an answer, run:

```sh
.venv/bin/python scripts/search.py "How robust are AI-text detectors to paraphrasing?"
```

## Rebuild extracted passages

If the PDFs or extraction rules change, rebuild chunks and reload the database:

```sh
.venv/bin/python scripts/build_chunks.py
.venv/bin/python scripts/ingest_database.py
```
