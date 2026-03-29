from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


_TOP_LEVEL_HISTORY_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")
_CHANNEL_HINT = re.compile(r"\b(cli|discord|telegram|whatsapp|api)\b", re.IGNORECASE)


@dataclass
class ChunkingStats:
    documents: int
    chunks: int
    history_documents: int
    markdown_documents: int


class DataModule:
    """Prepare source docs for RAG with parent-child chunk mapping."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        history_event_window: int = 1,
        history_event_overlap: int = 0,
        chunk_mode: str = "auto",
        recursive_chunk_size: int = 300,
        recursive_chunk_overlap: int = 20,
    ) -> None:
        self.data_path = Path(data_path)
        self.history_event_window = max(1, int(history_event_window))
        self.history_event_overlap = max(0, int(history_event_overlap))
        self.chunk_mode = (chunk_mode or "auto").strip().lower()
        self.recursive_chunk_size = max(50, int(recursive_chunk_size))
        self.recursive_chunk_overlap = max(0, int(recursive_chunk_overlap))

        self.documents: list[Document] = []
        self.chunks: list[Document] = []
        self.parent_child_map: dict[str, str] = {}
        self.parent_docs_by_id: dict[str, Document] = {}

    def load_documents(self) -> list[Document]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Path not found: {self.data_path}")

        files: list[Path]
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            files = sorted(self.data_path.rglob("*.md"))

        docs: list[Document] = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            parent_id = str(uuid4())
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "parent_id": parent_id,
                    "doc_type": "parent",
                },
            )
            self._enhance_metadata(doc)
            docs.append(doc)
            self.parent_docs_by_id[parent_id] = doc

        self.documents = docs
        return docs

    def _enhance_metadata(self, doc: Document) -> None:
        source = str(doc.metadata.get("source", ""))
        path = Path(source)

        is_history = path.name.lower() == "history.md"
        doc.metadata["is_history"] = is_history
        doc.metadata["category"] = "memory_history" if is_history else "markdown_doc"

        match = _CHANNEL_HINT.search(source)
        if match:
            doc.metadata["channel_hint"] = match.group(1).lower()

    def chunk_documents(
        self,
        *,
        markdown_headers: list[tuple[str, str]] | None = None,
        strip_headers: bool = False,
        chunk_mode: str | None = None,
    ) -> list[Document]:
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

        mode = (chunk_mode or self.chunk_mode).strip().lower()
        headers_to_split_on = markdown_headers or [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        chunks: list[Document] = []
        for doc in self.documents:
            if mode == "recursive_300":
                child_docs = self._recursive_split(
                    doc,
                    chunk_size=self.recursive_chunk_size,
                    chunk_overlap=self.recursive_chunk_overlap,
                )
            elif bool(doc.metadata.get("is_history")):
                child_docs = self._event_split_history(
                    doc,
                    event_window=self.history_event_window,
                    event_overlap=self.history_event_overlap,
                )
            else:
                child_docs = self._markdown_header_split(
                    doc,
                    headers_to_split_on=headers_to_split_on,
                    strip_headers=strip_headers,
                )
            chunks.extend(child_docs)

        self.chunks = chunks
        return chunks

    def _recursive_split(
        self,
        parent_doc: Document,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(50, int(chunk_size)),
            chunk_overlap=max(0, int(chunk_overlap)),
        )
        split_texts = splitter.split_text(parent_doc.page_content or "")
        out: list[Document] = []

        for i, text in enumerate(split_texts):
            child_id = str(uuid4())
            meta = dict(parent_doc.metadata)
            meta.update(
                {
                    "doc_type": "child",
                    "chunk_type": "recursive_300",
                    "chunk_id": child_id,
                    "chunk_index": i,
                    "chunk_size": len(text),
                }
            )
            out.append(Document(page_content=text, metadata=meta))
            self.parent_child_map[child_id] = str(parent_doc.metadata["parent_id"])
        return out

    def _markdown_header_split(
        self,
        parent_doc: Document,
        *,
        headers_to_split_on: list[tuple[str, str]],
        strip_headers: bool,
    ) -> list[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=strip_headers,
        )

        split_docs = splitter.split_text(parent_doc.page_content)
        out: list[Document] = []

        for i, chunk in enumerate(split_docs):
            child_id = str(uuid4())
            merged_meta = dict(parent_doc.metadata)
            merged_meta.update(dict(chunk.metadata or {}))
            merged_meta.update(
                {
                    "doc_type": "child",
                    "chunk_type": "markdown",
                    "chunk_id": child_id,
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                }
            )
            out.append(Document(page_content=chunk.page_content, metadata=merged_meta))
            self.parent_child_map[child_id] = str(parent_doc.metadata["parent_id"])
        return out

    def _event_split_history(
        self,
        parent_doc: Document,
        *,
        event_window: int,
        event_overlap: int,
    ) -> list[Document]:
        events = self._extract_history_events(parent_doc.page_content)
        if not events:
            child_id = str(uuid4())
            meta = dict(parent_doc.metadata)
            meta.update(
                {
                    "doc_type": "child",
                    "chunk_type": "event_fallback",
                    "chunk_id": child_id,
                    "chunk_index": 0,
                    "event_count": 1,
                    "chunk_size": len(parent_doc.page_content),
                }
            )
            self.parent_child_map[child_id] = str(parent_doc.metadata["parent_id"])
            return [Document(page_content=parent_doc.page_content, metadata=meta)]

        step = max(1, event_window - min(event_overlap, event_window - 1))
        chunks: list[Document] = []
        chunk_index = 0

        for start in range(0, len(events), step):
            window = events[start : start + event_window]
            if not window:
                continue

            text = "\n\n".join(item["text"] for item in window).strip()
            if not text:
                continue

            child_id = str(uuid4())
            channels = sorted({c for item in window for c in item["channels"]})
            timestamps = [item["timestamp"] for item in window if item["timestamp"]]

            meta = dict(parent_doc.metadata)
            meta.update(
                {
                    "doc_type": "child",
                    "chunk_type": "event",
                    "chunk_id": child_id,
                    "chunk_index": chunk_index,
                    "chunk_size": len(text),
                    "event_count": len(window),
                    "event_start_index": start,
                    "event_end_index": start + len(window) - 1,
                    "timestamp_start": timestamps[0] if timestamps else None,
                    "timestamp_end": timestamps[-1] if timestamps else None,
                    "channels": channels,
                }
            )

            chunks.append(Document(page_content=text, metadata=meta))
            self.parent_child_map[child_id] = str(parent_doc.metadata["parent_id"])
            chunk_index += 1

            if start + event_window >= len(events):
                break

        return chunks

    def _extract_history_events(self, content: str) -> list[dict[str, Any]]:
        lines = content.splitlines()
        starts: list[int] = []
        stamps: list[str] = []

        for i, line in enumerate(lines):
            match = _TOP_LEVEL_HISTORY_TS.match(line.strip())
            if not match:
                continue
            starts.append(i)
            stamps.append(match.group(1))

        if not starts:
            return []

        events: list[dict[str, Any]] = []
        for idx, line_index in enumerate(starts):
            next_index = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
            event_text = "\n".join(lines[line_index:next_index]).strip()
            if not event_text:
                continue

            channel_hits = sorted(
                {m.group(1).lower() for m in _CHANNEL_HINT.finditer(event_text)}
            )
            events.append(
                {
                    "timestamp": stamps[idx],
                    "text": event_text,
                    "channels": channel_hits,
                }
            )
        return events

    def get_parent_documents(self, child_chunks: list[Document]) -> list[Document]:
        hit_count: dict[str, int] = {}
        for chunk in child_chunks:
            parent_id = str(chunk.metadata.get("parent_id", "")).strip()
            if parent_id:
                hit_count[parent_id] = hit_count.get(parent_id, 0) + 1

        ranked_parent_ids = sorted(
            hit_count.keys(),
            key=lambda pid: hit_count[pid],
            reverse=True,
        )
        return [
            self.parent_docs_by_id[pid]
            for pid in ranked_parent_ids
            if pid in self.parent_docs_by_id
        ]

    def get_stats(self) -> ChunkingStats:
        history_docs = sum(1 for d in self.documents if bool(d.metadata.get("is_history")))
        markdown_docs = len(self.documents) - history_docs
        return ChunkingStats(
            documents=len(self.documents),
            chunks=len(self.chunks),
            history_documents=history_docs,
            markdown_documents=markdown_docs,
        )
