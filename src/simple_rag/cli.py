from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .rag import SimpleRAG


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="simpleRAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest plain text lines")
    ingest.add_argument("--input", required=True, help="Path to a text file")
    ingest.add_argument("--source", default="local", help="Metadata source value")

    query = subparsers.add_parser("query", help="Query top-k similar docs")
    query.add_argument("--question", required=True, help="User question")
    query.add_argument("--top-k", type=int, default=3, help="How many docs to return")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    rag = SimpleRAG(load_settings())

    if args.command == "ingest":
        input_path = Path(args.input)
        added = rag.ingest_texts(_read_lines(input_path), source=args.source)
        print(f"Ingested {added} documents from {input_path}")
        return

    if args.command == "query":
        hits = rag.query(question=args.question, top_k=args.top_k)
        if not hits:
            print("No results found.")
            return

        for index, hit in enumerate(hits, start=1):
            print(f"[{index}] distance={hit['distance']:.4f}")
            print(hit["text"])
            print(f"metadata={hit['metadata']}")
            print()


if __name__ == "__main__":
    main()
