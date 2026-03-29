from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    chroma_dir: str = ".chroma"
    chroma_collection: str = "simple_rag_docs"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        embed_model=os.getenv("EMBED_MODEL", Settings.embed_model),
        chroma_dir=os.getenv("CHROMA_DIR", Settings.chroma_dir),
        chroma_collection=os.getenv("CHROMA_COLLECTION", Settings.chroma_collection),
    )
