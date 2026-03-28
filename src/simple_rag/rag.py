from __future__ import annotations

from typing import Iterable
from uuid import uuid4

import chromadb
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


from .config import Settings


class SimpleRAG:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = SentenceTransformer(settings.embed_model)
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._collection = client.get_or_create_collection(name=settings.chroma_collection)

    # loop






