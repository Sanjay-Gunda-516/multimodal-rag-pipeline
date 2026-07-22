# config.py
"""
Central configuration for the Multimodal RAG pipeline.

All model names, paths, and tuning parameters live here.
Swap any model by changing a single string — no hunting through files.

Pydantic BaseSettings auto-loads values from .env and validates
types at startup. If a required key is missing, the app crashes
immediately with a clear error — not silently 10 seconds later.
"""

from functools import lru_cache
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Pydantic Settings Class ────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All settings auto-loaded from environment variables or .env file.
    Priority order: env vars > .env file > field defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # which file to read
        env_file_encoding="utf-8", # encoding of that file
        case_sensitive=False,      # ANTHROPIC_API_KEY == anthropic_api_key
        extra="ignore",            # ignore unknown keys in .env silently
    )

    # ── API Keys ───────────────────────────────────────────────────────────────
    # Field(...) means REQUIRED — app crashes at startup if missing
    anthropic_api_key: str = Field(..., description="Anthropic API key")

    # Field("") means OPTIONAL — defaults to empty string if not set
    langfuse_public_key: str = Field("", description="Langfuse public key")
    langfuse_secret_key: str = Field("", description="Langfuse secret key")
    langfuse_host: str = Field(
        "https://cloud.langfuse.com", description="Langfuse host URL"
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    temperature: float = 0.1   # low = more factual, less creative — correct for RAG

    # ── Text Embedding Model ───────────────────────────────────────────────────
    # To benchmark a different model: change this one string
    text_embedding_model: str = "all-MiniLM-L6-v2"

    # ── CLIP (Vision Embedding) ────────────────────────────────────────────────
    clip_model: str = "ViT-B-32"       # architecture
    clip_pretrained: str = "openai"    # which pretrained weights to load

    # ── Reranker ──────────────────────────────────────────────────────────────
    # To benchmark: swap to "BAAI/bge-reranker-v2-m3" and re-run eval
    reranker_model: str = "BAAI/bge-reranker-base"

    # ── Hallucination Scorer ──────────────────────────────────────────────────
    nli_model: str = "cross-encoder/nli-deberta-v3-small"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 500    # tokens per chunk
    chunk_overlap: int = 50  # overlap between consecutive chunks — prevents lost context

    # ── Retrieval ─────────────────────────────────────────────────────────────
    bm25_top_k: int = 20     # BM25 returns this many candidates
    vector_top_k: int = 20   # Vector search returns this many candidates
    reranker_top_k: int = 5  # Re-ranker keeps the best N from the merged 40
    image_top_k: int = 2     # Number of relevant images passed to the LLM

    # ── ChromaDB Collection Names ─────────────────────────────────────────────
    text_collection_name: str = "text_chunks"
    image_collection_name: str = "image_embeddings"


# ── Derived Directory Paths ────────────────────────────────────────────────────
# These come from the project layout, not from .env

ROOT_DIR   = Path(__file__).parent          # where config.py lives = project root
DATA_DIR   = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"          # uploaded PDFs go here
IMAGES_DIR  = DATA_DIR / "images"           # extracted images go here
CHROMA_DIR  = ROOT_DIR / "chroma_db"        # ChromaDB persists here
BM25_DIR    = ROOT_DIR / "bm25s_index"      # BM25 index persists here
LOGS_DIR    = ROOT_DIR / "logs"             # loguru writes here


# ── Apple Silicon Device Selection ────────────────────────────────────────────

def get_device() -> str:
    """
    Returns the best available compute device.

    Priority: MPS (Apple Silicon GPU) > CUDA (NVIDIA) > CPU
    All our embedding and reranker models read this at load time.
    """
    if torch.backends.mps.is_available():
        return "mps"   # M1 / M2 / M3 — Metal Performance Shaders
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU (not applicable on your machine, but good practice)
    return "cpu"


# ── Directory Bootstrap ───────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """
    Creates all required directories if they don't already exist.
    Call this once at application startup before any file I/O.
    """
    for directory in [UPLOADS_DIR, IMAGES_DIR, CHROMA_DIR, BM25_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ── Singleton Pattern ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the same Settings instance every time.

    @lru_cache(maxsize=1) means: cache the result of the first call
    and return it for every subsequent call. The .env file is only
    read once, no matter how many modules import get_settings().

    In tests, call get_settings.cache_clear() to reset between tests.
    """
    return Settings()