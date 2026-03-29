"""
RAG pipeline and generation integration module.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable

from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .data_module import DataModule
from .index_module import IndexModule
from .retrieve_module import RetrieveModule

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """Generic generation module for memory-oriented RAG."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = self._build_llm()

    _TS_PATTERNS = (
        re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"),
        re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}"),
    )

    @classmethod
    def _should_skip_rewrite(cls, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return True
        if len(q) <= 12:
            return True
        if any(p.search(q) for p in cls._TS_PATTERNS):
            return True
        # Keep mixed English entity names stable (e.g., "NLP and deep learning")
        if re.search(r"[A-Za-z]{2,}", q) and len(q) <= 48:
            return True
        return False

    def _build_llm(self) -> ChatOpenAI:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY environment variable.")

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        logger.info("Initializing LLM: %s", self.model_name)
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

    def _build_context(self, docs: list[Document], max_length: int = 3000) -> str:
        if not docs:
            return "暂无相关记忆信息。"

        parts: list[str] = []
        current_len = 0
        for i, doc in enumerate(docs, start=1):
            meta = dict(doc.metadata or {})
            header = [f"【文档 {i}】"]
            if "title" in meta:
                header.append(str(meta["title"]))
            if "category" in meta:
                header.append(f"分类: {meta['category']}")
            if "timestamp_start" in meta and meta["timestamp_start"]:
                header.append(f"起始时间: {meta['timestamp_start']}")
            if "timestamp_end" in meta and meta["timestamp_end"]:
                header.append(f"结束时间: {meta['timestamp_end']}")
            if "file_name" in meta:
                header.append(f"文件: {meta['file_name']}")
            if "source" in meta:
                header.append(f"来源: {meta['source']}")

            block = " | ".join(header) + "\n" + doc.page_content.strip() + "\n"
            if current_len + len(block) > max_length:
                break
            parts.append(block)
            current_len += len(block)
        return "\n" + ("=" * 50) + "\n".join(parts)

    def query_rewrite(self, query: str) -> str:
        if self._should_skip_rewrite(query):
            return query

        prompt = PromptTemplate(
            template=(
                "你是通用查询重写助手。若原问题已经具体清晰，则保持不变；"
                "若过于口语或模糊，重写为更利于记忆检索的简洁查询。\n\n"
                "原始查询: {query}\n\n输出最终查询："
            ),
            input_variables=["query"],
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke(query).strip()
        if rewritten and rewritten != query:
            logger.info("Query rewritten: '%s' -> '%s'", query, rewritten)
        return rewritten
        return query

    def query_router(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "将用户问题分类为 list/detail/general 三种之一，仅返回分类词。\n"
            "list: 用户要点清单/候选列表；detail: 用户要深入解释；general: 其他。\n\n"
            "用户问题: {query}\n分类结果:"
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        route = chain.invoke(query).strip().lower()
        return route if route in {"list", "detail", "general"} else "general"

    def generate_list_answer(self, query: str, context_docs: list[Document]) -> str:
        if not context_docs:
            return "抱歉，没有找到相关信息。"

        names: list[str] = []
        for d in context_docs:
            meta = dict(d.metadata or {})
            candidate = (
                meta.get("title")
                or meta.get("file_name")
                or meta.get("source")
                or meta.get("timestamp_start")
            )
            if isinstance(candidate, str) and candidate and candidate not in names:
                names.append(candidate)

        if not names:
            return self.generate_basic_answer(query, context_docs)

        if len(names) <= 3:
            return "为你整理到以下相关条目：\n" + "\n".join(
                f"{i + 1}. {name}" for i, name in enumerate(names)
            )
        return "为你整理到以下相关条目：\n" + "\n".join(
            f"{i + 1}. {name}" for i, name in enumerate(names[:3])
        )

    def generate_basic_answer(self, query: str, context_docs: list[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            "你是一个通用 RAG 助手。请基于检索到的记忆/资料回答用户问题。\n\n"
            "用户问题: {question}\n\n检索上下文:\n{context}\n\n"
            "要求：\n"
            "1. 优先依据上下文作答，不要臆造；\n"
            "2. 结论尽量清晰；\n"
            "3. 如果上下文不足，明确说明缺失信息。"
        )
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_step_by_step_answer(self, query: str, context_docs: list[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            "你是一个通用 RAG 助手。请基于上下文给出分步骤分析。\n\n"
            "用户问题: {question}\n\n检索上下文:\n{context}\n\n"
            "可按以下结构组织：\n"
            "1) 关键信息提取\n2) 推理过程\n3) 最终结论\n4) 不确定点/补充建议\n"
            "要求：不得编造上下文外事实。"
        )
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_basic_answer_stream(self, query: str, context_docs: list[Document]):
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            "你是一个通用 RAG 助手。请根据上下文回答问题。\n\n"
            "用户问题: {question}\n\n检索上下文:\n{context}\n\n回答:"
        )
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        yield from chain.stream(query)

    def generate_step_by_step_answer_stream(self, query: str, context_docs: list[Document]):
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            "你是一个通用 RAG 助手。请根据上下文给出分步骤分析。\n\n"
            "用户问题: {question}\n\n检索上下文:\n{context}\n\n回答:"
        )
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        yield from chain.stream(query)


class SimpleRAG:
    """Composable RAG pipeline built from Data/Index/Retrieve/Generation modules."""

    def __init__(
        self,
        *,
        data_path: str,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_device: str = "auto",
        llm_model: str = "deepseek-chat",
        index_save_path: str = "./index_store",
        history_event_window: int = 1,
        history_event_overlap: int = 0,
        chunk_mode: str = "auto",
    ) -> None:
        self.data_module = DataModule(
            data_path,
            history_event_window=history_event_window,
            history_event_overlap=history_event_overlap,
            chunk_mode=chunk_mode,
        )
        self.index_module = IndexModule(
            model_name=embedding_model,
            index_save_path=index_save_path,
            device=embedding_device,
        )
        self.gen_module = GenerationIntegrationModule(model_name=llm_model)
        self.retrieve_module: RetrieveModule | None = None

    @staticmethod
    def _route_override(query: str) -> str | None:
        q = (query or "").strip().lower()
        if not q:
            return None

        detail_keywords = [
            "聊过什么",
            "页面有什么",
            "内容",
            "详情",
            "是什么",
            "总结",
            "解释",
        ]
        list_keywords = [
            "推荐",
            "列表",
            "列出",
            "有哪些",
            "有什么可选",
        ]

        if any(k in q for k in detail_keywords):
            return "detail"
        if any(k in q for k in list_keywords):
            return "list"
        return None

    def ingest(self) -> dict:
        self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()
        self.index_module.build_index(chunks)
        self.index_module.save_index()
        self.retrieve_module = RetrieveModule(self.index_module.vectorstore, chunks)
        stats = self.data_module.get_stats()
        return {
            "documents": stats.documents,
            "chunks": stats.chunks,
            "history_documents": stats.history_documents,
            "markdown_documents": stats.markdown_documents,
        }

    def load_index(self) -> bool:
        vectorstore = self.index_module.load_index()
        if vectorstore is None:
            return False

        self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()
        self.retrieve_module = RetrieveModule(vectorstore, chunks)
        return True

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        if self.retrieve_module is None:
            raise ValueError("Retriever is not ready. Run ingest() or load_index() first.")
        rewritten = self.gen_module.query_rewrite(query)
        return self.retrieve_module.hybrid_retrieve(rewritten, k=top_k)

    def retrieve_with_debug(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if self.retrieve_module is None:
            raise ValueError("Retriever is not ready. Run ingest() or load_index() first.")
        rewritten = self.gen_module.query_rewrite(query)
        chunks = self.retrieve_module.hybrid_retrieve(rewritten, k=top_k)
        parent_docs = self.data_module.get_parent_documents(chunks)
        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "chunks": chunks,
            "parent_docs": parent_docs,
        }

    def answer_from_chunks(
        self,
        query: str,
        chunks: list[Document],
        *,
        use_step_by_step: bool = False,
    ) -> str:
        # For QA reliability, prefer retrieved chunks directly.
        # Parent documents can be too long/noisy for focused answers.
        context_docs = chunks

        route = self._route_override(query) or self.gen_module.query_router(query)
        if route == "list":
            return self.gen_module.generate_list_answer(query, context_docs)
        if route == "detail" or use_step_by_step:
            return self.gen_module.generate_step_by_step_answer(query, context_docs)
        return self.gen_module.generate_basic_answer(query, context_docs)

    def answer(self, query: str, top_k: int = 5, use_step_by_step: bool = False) -> str:
        chunks = self.retrieve(query, top_k=top_k)
        return self.answer_from_chunks(query, chunks, use_step_by_step=use_step_by_step)

    def answer_stream(
        self,
        query: str,
        top_k: int = 5,
        use_step_by_step: bool = False,
    ) -> Iterable[str]:
        chunks = self.retrieve(query, top_k=top_k)
        context_docs = chunks

        route = self.gen_module.query_router(query)
        if route == "detail" or use_step_by_step:
            yield from self.gen_module.generate_step_by_step_answer_stream(query, context_docs)
            return
        yield from self.gen_module.generate_basic_answer_stream(query, context_docs)
