from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .rag import SimpleRAG


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _preview_text(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _format_chunk_rows(chunks: list[Any]) -> str:
    if not chunks:
        return "No relevant chunks found."

    lines: list[str] = []
    for idx, doc in enumerate(chunks, start=1):
        meta = dict(getattr(doc, "metadata", {}) or {})
        source = meta.get("source") or meta.get("file_name") or "(unknown source)"
        chunk_type = meta.get("chunk_type", "")
        rrf_score = meta.get("rrf_score", "")
        preview = _preview_text(getattr(doc, "page_content", ""))

        extras: list[str] = []
        if chunk_type:
            extras.append(f"type={chunk_type}")
        if rrf_score != "":
            extras.append(f"rrf={rrf_score}")
        suffix = f" {' '.join(extras)}" if extras else ""
        lines.append(f"[{idx}] source={source}{suffix}\n{preview}")
    return "\n\n".join(lines)


class SimpleRAGRuntime:
    def __init__(
        self,
        *,
        data_path: str,
        index_path: str,
        embedding_model: str,
        embedding_device: str,
        llm_model: str,
        chunk_mode: str,
    ) -> None:
        self._data_path = data_path
        self._index_path = index_path
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device
        self._llm_model = llm_model
        self._chunk_mode = chunk_mode
        self._rag: SimpleRAG | None = None

    def _build(self) -> SimpleRAG:
        if self._rag is None:
            self._rag = SimpleRAG(
                data_path=self._data_path,
                embedding_model=self._embedding_model,
                embedding_device=self._embedding_device,
                llm_model=self._llm_model,
                index_save_path=self._index_path,
                chunk_mode=self._chunk_mode,
            )
        return self._rag

    def ensure_ready(self, *, force_reindex: bool = False) -> SimpleRAG:
        rag = self._build()
        if force_reindex:
            rag.ingest()
            return rag
        if not rag.load_index():
            rag.ingest()
        return rag

    def ingest(self) -> dict[str, Any]:
        rag = self._build()
        return rag.ingest()


class SimpleRAGRuntimeHub:
    def __init__(
        self,
        *,
        default_data_path: str,
        default_index_path: str,
        embedding_model: str,
        embedding_device: str,
        llm_model: str,
        chunk_mode: str,
    ) -> None:
        self._default_data_path = default_data_path
        self._default_index_path = default_index_path
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device
        self._llm_model = llm_model
        self._chunk_mode = chunk_mode
        self._runtimes: dict[tuple[str, str], SimpleRAGRuntime] = {}

    def get(
        self,
        *,
        data_path: str | None = None,
        index_path: str | None = None,
    ) -> SimpleRAGRuntime:
        resolved_data_path = str(Path(data_path or self._default_data_path).expanduser().resolve())
        resolved_index_path = str(Path(index_path or self._default_index_path).expanduser().resolve())
        key = (resolved_data_path, resolved_index_path)
        runtime = self._runtimes.get(key)
        if runtime is None:
            runtime = SimpleRAGRuntime(
                data_path=resolved_data_path,
                index_path=resolved_index_path,
                embedding_model=self._embedding_model,
                embedding_device=self._embedding_device,
                llm_model=self._llm_model,
                chunk_mode=self._chunk_mode,
            )
            self._runtimes[key] = runtime
        return runtime


def create_mcp_server(
    *,
    name: str = "simple-rag",
    data_path: str | None = None,
    index_path: str | None = None,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedding_device: str = "auto",
    llm_model: str = "deepseek-chat",
    chunk_mode: str = "auto",
):
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("MCP SDK is not installed. Install it with: pip install mcp") from exc

    load_dotenv(PROJECT_ROOT / ".env")

    runtime_hub = SimpleRAGRuntimeHub(
        default_data_path=data_path or os.getenv("RAG_DATA_PATH", str(PROJECT_ROOT / "data")),
        default_index_path=index_path or os.getenv("RAG_INDEX_PATH", str(PROJECT_ROOT / "index_store")),
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        llm_model=llm_model,
        chunk_mode=chunk_mode,
    )

    mcp = FastMCP(name=name)

    @mcp.tool(
        name="rag_ingest",
        description="Build or rebuild the simpleRAG index from the configured data path",
    )
    async def rag_ingest(data_path: str = "", index_path: str = "") -> str:
        runtime = runtime_hub.get(data_path=data_path or None, index_path=index_path or None)
        stats = runtime.ingest()
        return (
            "simpleRAG index ready.\n"
            f"documents={stats.get('documents', 0)} chunks={stats.get('chunks', 0)} "
            f"history_documents={stats.get('history_documents', 0)} "
            f"markdown_documents={stats.get('markdown_documents', 0)}"
        )

    @mcp.tool(
        name="rag_retrieve",
        description="Retrieve top-k relevant chunks from simpleRAG without generating an answer",
    )
    async def rag_retrieve(
        query: str,
        top_k: int = 5,
        reindex: bool = False,
        data_path: str = "",
        index_path: str = "",
    ) -> str:
        rag = runtime_hub.get(
            data_path=data_path or None,
            index_path=index_path or None,
        ).ensure_ready(force_reindex=reindex)
        debug = rag.retrieve_with_debug(query, top_k=top_k)
        chunks = debug.get("chunks", [])
        header = (
            f"original_query={debug.get('original_query', '')}\n"
            f"rewritten_query={debug.get('rewritten_query', '')}\n"
            f"chunks={len(chunks)}"
        )
        body = _format_chunk_rows(chunks)
        return f"{header}\n\n{body}"

    @mcp.tool(
        name="rag_answer",
        description="Answer a question with simpleRAG using retrieval plus generation",
    )
    async def rag_answer(
        query: str,
        top_k: int = 5,
        step_by_step: bool = False,
        reindex: bool = False,
        data_path: str = "",
        index_path: str = "",
    ) -> str:
        rag = runtime_hub.get(
            data_path=data_path or None,
            index_path=index_path or None,
        ).ensure_ready(force_reindex=reindex)
        return rag.answer(query, top_k=top_k, use_step_by_step=step_by_step)

    return mcp


def run_mcp_server(
    *,
    transport: str = "stdio",
    name: str = "simple-rag",
    data_path: str | None = None,
    index_path: str | None = None,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedding_device: str = "auto",
    llm_model: str = "deepseek-chat",
    chunk_mode: str = "auto",
) -> None:
    mcp = create_mcp_server(
        name=name,
        data_path=data_path,
        index_path=index_path,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        llm_model=llm_model,
        chunk_mode=chunk_mode,
    )
    mcp.run(transport=transport)
