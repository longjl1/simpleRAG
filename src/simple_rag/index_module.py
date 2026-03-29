from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class IndexModule:
    """Embedding + Chroma index build/load/save."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        index_save_path: str = "./index_store",
        device: str = "auto",
        collection_name: str = "simple_rag_docs",
    ) -> None:
        self.model_name = model_name
        self.index_save_path = str(Path(index_save_path).resolve())
        self.device = device
        self.collection_name = collection_name
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vectorstore: Chroma | None = None
        self.setup_embeddings()

    @staticmethod
    def _resolve_device(device: str) -> str:
        request = (device or "auto").strip().lower()
        if request in {"cpu", "cuda"}:
            return request

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def setup_embeddings(self) -> None:
        resolved = self._resolve_device(self.device)
        logger.info("Embedding model=%s device=%s", self.model_name, resolved)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": resolved},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _open_chroma(self) -> Chroma:
        if self.embeddings is None:
            self.setup_embeddings()
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.index_save_path,
        )

    def build_index(
        self,
        documents: Iterable[Document],
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> Chroma:
        docs = list(documents)
        if not docs:
            raise ValueError("No documents provided for index construction.")

        if embeddings is not None:
            self.embeddings = embeddings
        if self.embeddings is None:
            self.setup_embeddings()

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        vectorstore = self._open_chroma()
        vectorstore.add_documents(docs)
        self.vectorstore = vectorstore
        return vectorstore

    def save_index(self) -> None:
        if self.vectorstore is None:
            raise ValueError("No index to save. Please build the index first.")
        persist_fn = getattr(self.vectorstore, "persist", None)
        if callable(persist_fn):
            persist_fn()

    def load_index(self) -> Chroma | None:
        index_path = Path(self.index_save_path)
        if not index_path.exists():
            return None

        self.vectorstore = self._open_chroma()
        # Return None when the collection exists but is empty, so caller can ingest.
        try:
            count = self.vectorstore._collection.count()  # type: ignore[attr-defined]
            if count <= 0:
                return None
        except Exception:
            # If count probing fails, still return the opened vectorstore.
            pass
        return self.vectorstore
