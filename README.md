# simpleRAG

A minimal RAG scaffold using:

- `sentence-transformers` for embeddings
- `chromadb` for vector storage
- `langchain` for document objects

## Quick start

```bash
uv sync
copy .env.example .env
uv run simple-rag ingest --input data/sample_docs.txt
uv run simple-rag query --question "What is simpleRAG?"
```

## Project structure

```text
simpleRAG/
  data/
    sample_docs.txt
  src/
    simple_rag/
      __init__.py
      cli.py
      config.py
      rag.py
```

## Notes

- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Default local Chroma path: `./.chroma`
