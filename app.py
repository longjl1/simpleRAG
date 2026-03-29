from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simple_rag.rag import SimpleRAG  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="simpleRAG app")
    parser.add_argument(
        "--data-path",
        default=os.getenv("RAG_DATA_PATH", str(PROJECT_ROOT / "data")),
        help="Data file or directory path (default: env RAG_DATA_PATH or ./data)",
    )
    parser.add_argument(
        "--index-path",
        default=os.getenv("RAG_INDEX_PATH", str(PROJECT_ROOT / "index_store")),
        help="Chroma persistence directory",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-device",
        default=os.getenv("EMBED_DEVICE", "auto"),
        choices=["auto", "cpu", "cuda"],
        help="Embedding device",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LLM_MODEL", "deepseek-chat"),
        help="LLM model name",
    )
    parser.add_argument(
        "--chunk-mode",
        default=os.getenv("CHUNK_MODE", "auto"),
        choices=["auto", "recursive_300"],
        help="Chunk strategy: auto (history event split + markdown split) or recursive_300",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval")
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Print retrieval debug info (rewritten query, chunk metadata, previews)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild index before answering",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Force step-by-step answer format",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build and save index")

    ask = subparsers.add_parser("ask", help="Ask one question")
    ask.add_argument("query", help="User query")

    chat = subparsers.add_parser("chat", help="Interactive chat loop")
    chat.add_argument(
        "--exit-words",
        nargs="*",
        default=["exit", "quit", "q"],
        help="Words to stop interactive chat",
    )

    return parser


def build_rag(args: argparse.Namespace) -> SimpleRAG:
    return SimpleRAG(
        data_path=args.data_path,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        llm_model=args.llm_model,
        index_save_path=args.index_path,
        chunk_mode=args.chunk_mode,
    )


def ensure_index(rag: SimpleRAG, force_reindex: bool = False) -> None:
    if force_reindex:
        stats = rag.ingest()
        logging.info("Re-index complete: %s", stats)
        return

    if rag.load_index():
        logging.info("Loaded existing index.")
        return

    stats = rag.ingest()
    logging.info("Index built: %s", stats)


def run_ingest(args: argparse.Namespace) -> int:
    rag = build_rag(args)
    stats = rag.ingest()
    print("Index built successfully.")
    print(stats)
    return 0


def run_ask(args: argparse.Namespace) -> int:
    rag = build_rag(args)
    ensure_index(rag, force_reindex=args.reindex)
    if args.debug_retrieval:
        debug = rag.retrieve_with_debug(args.query, top_k=args.top_k)
        _print_retrieval_debug(debug)
        answer = rag.answer_from_chunks(
            args.query,
            debug["chunks"],
            use_step_by_step=args.step,
        )
    else:
        answer = rag.answer(args.query, top_k=args.top_k, use_step_by_step=args.step)
    print(answer)
    return 0


def run_chat(args: argparse.Namespace) -> int:
    rag = build_rag(args)
    ensure_index(rag, force_reindex=args.reindex)
    exit_words = {w.lower() for w in args.exit_words}

    print("simpleRAG chat started. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            return 0

        if not query:
            continue
        if query.lower() in exit_words:
            print("Bye.")
            return 0

        try:
            if args.debug_retrieval:
                debug = rag.retrieve_with_debug(query, top_k=args.top_k)
                _print_retrieval_debug(debug)
                answer = rag.answer_from_chunks(
                    query,
                    debug["chunks"],
                    use_step_by_step=args.step,
                )
            else:
                answer = rag.answer(query, top_k=args.top_k, use_step_by_step=args.step)
        except Exception as exc:
            print(f"Error: {exc}")
            continue
        print(f"\nRAG> {answer}")


def _preview_text(text: str, limit: int = 180) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _print_retrieval_debug(debug: dict) -> None:
    original_query = debug.get("original_query", "")
    rewritten_query = debug.get("rewritten_query", "")
    chunks = debug.get("chunks", [])
    parents = debug.get("parent_docs", [])

    print("\n[Retrieval Debug]")
    print(f"- original_query: {original_query}")
    print(f"- rewritten_query: {rewritten_query}")
    print(f"- chunks: {len(chunks)}")
    print(f"- parent_docs: {len(parents)}")

    for idx, doc in enumerate(chunks, start=1):
        meta = dict(getattr(doc, "metadata", {}) or {})
        source = meta.get("source", "")
        chunk_type = meta.get("chunk_type", "")
        ts_start = meta.get("timestamp_start", "")
        ts_end = meta.get("timestamp_end", "")
        rrf = meta.get("rrf_score", "")
        print(f"  [{idx}] source={source} chunk_type={chunk_type} ts={ts_start}->{ts_end} rrf={rrf}")
        print(f"      {_preview_text(getattr(doc, 'page_content', ''))}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "ask":
        return run_ask(args)
    if args.command == "chat":
        return run_chat(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
