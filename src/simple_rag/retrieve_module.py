from __future__ import annotations

import logging
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrieveModule:
    """Hybrid retrieval module: vector search + optional BM25 + RRF rerank."""

    def __init__(self, vectorstore, chunks: list[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_retriever = None
        self.bm25_retriever = None
        self.setup_retrievers()

    def setup_retrievers(self) -> None:
        logger.info("Initializing retrievers...")
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        try:
            self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)
        except ImportError as exc:
            logger.warning(
                "BM25 unavailable (rank_bm25 not installed). Fallback to vector-only retrieval. details=%s",
                exc,
            )
            self.bm25_retriever = None

    def hybrid_retrieve(self, query: str, k: int = 3) -> list[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        if self.bm25_retriever is None:
            return vector_docs[:k]

        bm25_docs = self.bm25_retriever.invoke(query)
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:k]

    def metadata_filtered_retreieve(
        self, query: str, metadata_filters: dict[str, Any], k: int = 3
    ) -> list[Document]:
        docs = self.hybrid_retrieve(query, k=10)
        filtered_docs: list[Document] = []

        for doc in docs:
            match = True
            for key, value in metadata_filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                else:
                    if doc.metadata[key] != value:
                        match = False
                        break

            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= k:
                    break
        return filtered_docs

    def metadata_filtered_retrieve(
        self, query: str, metadata_filters: dict[str, Any], k: int = 3
    ) -> list[Document]:
        return self.metadata_filtered_retreieve(query, metadata_filters, k=k)

    def _rrf_rerank(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        k: int = 60,
    ) -> list[Document]:
        doc_scores: dict[int, float] = {}
        doc_objects: dict[int, Document] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # add score
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_docs: list[Document] = []
        for doc_id, final_score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = float(final_score)
            reranked_docs.append(doc)
        return reranked_docs
