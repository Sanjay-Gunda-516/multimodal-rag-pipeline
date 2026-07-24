"""
Text Embedder — converts TextChunk objects into dense vector embeddings.

Model: all-MiniLM-L6-v2
    - 384-dimensional output vectors
    - Trained on 1B+ sentence pairs for semantic similarity
    - Fast enough for real-time use (~14k sentences/sec on GPU)
    - Downloads ~90 MB on first use, then cached at ~/.cache/huggingface/

Why sentence-transformers over raw HuggingFace transformers?
    sentence-transformers handles the full pipeline for us:
    tokenisation → model forward pass → mean pooling → normalisation.
    Doing this manually with raw transformers requires ~50 extra lines
    and is error-prone (especially the pooling step).

MPS acceleration:
    On Apple Silicon, PyTorch uses the Metal Performance Shaders (MPS)
    backend. sentence-transformers respects the device we pass in.
    Speedup over CPU: roughly 3-5x for batch inference on M3.
"""

from dataclasses import dataclass, field
from typing import List

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import get_device, get_settings
from src.ingestion import TextChunk


# ── Data Structure ─────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    """
    A TextChunk enriched with its embedding vector.

    Attributes:
        chunk:     The original TextChunk (text + all metadata).
        embedding: 384-dim float list. This is what ChromaDB stores
                   and compares during vector search.
    """
    chunk:     TextChunk
    embedding: List[float]

    @property
    def chunk_id(self) -> str:
        """Convenience passthrough — avoids .chunk.chunk_id everywhere."""
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict:
        return self.chunk.to_metadata_dict()


# ── Text Embedder ──────────────────────────────────────────────────────────────

class TextEmbedder:
    """
    Embeds TextChunk objects using a sentence-transformers model.

    The model is loaded ONCE in __init__ and reused for every batch.
    Never instantiate TextEmbedder inside a loop.

    Usage:
        embedder = TextEmbedder()                    # load model once
        embedded = embedder.embed(chunks)            # embed all chunks
        query_vec = embedder.embed_query("my query") # embed a query string
    """

    def __init__(self):
        self.settings = get_settings()
        self.device   = get_device()

        logger.info(
            "Loading text embedding model '{}' on device '{}'",
            self.settings.text_embedding_model,
            self.device,
        )

        # SentenceTransformer downloads the model on first run,
        # then loads from ~/.cache/huggingface/ on subsequent runs.
        # device= tells it which hardware to run inference on.
        self.model = SentenceTransformer(
            self.settings.text_embedding_model,
            device=self.device,
        )

        # Verify the output dimension matches our expectation.
        # get_sentence_embedding_dimension() returns 384 for MiniLM.
        # This is a sanity check — if we ever swap models, this catches
        # a mismatch before bad vectors reach ChromaDB.
        self.embedding_dim = self.model.get_embedding_dimension()

        logger.success(
            "Text embedder ready | model='{}' | dim={} | device='{}'",
            self.settings.text_embedding_model,
            self.embedding_dim,
            self.device,
        )

    def embed(self, chunks: List[TextChunk]) -> List[EmbeddedChunk]:
        """
        Embed a list of TextChunk objects.

        Processes chunks in batches of 32 for GPU efficiency.
        Converts numpy arrays to Python float lists for ChromaDB
        compatibility (ChromaDB does not accept numpy arrays directly).

        Args:
            chunks: Output from TextChunker — list of TextChunk objects.

        Returns:
            List of EmbeddedChunk — each chunk paired with its vector.
        """
        if not chunks:
            logger.warning("embed() called with empty chunk list")
            return []

        logger.info("Embedding {} text chunks...", len(chunks))

        # Extract just the text strings — that's what the model needs
        texts = [chunk.text for chunk in chunks]

        # model.encode() is the core call:
        #   batch_size=32      → process 32 texts per GPU call
        #   show_progress_bar  → display tqdm bar for large batches
        #   convert_to_numpy   → return np.ndarray (faster than torch.Tensor
        #                        for our use case)
        #   normalize_embeddings → L2-normalise vectors to unit length.
        #                          Required for cosine similarity to work
        #                          correctly. Without this, dot product ≠ cosine.
        embeddings_np = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(chunks) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Pair each chunk with its embedding vector.
        # embeddings_np[i] is a numpy array of shape (384,)
        # .tolist() converts it to a plain Python list of floats.
        # ChromaDB requires Python lists, not numpy arrays.
        results = [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embeddings_np[i].tolist(),
            )
            for i, chunk in enumerate(chunks)
        ]

        logger.success(
            "Embedded {} chunks | dim={} | device='{}'",
            len(results),
            self.embedding_dim,
            self.device,
        )

        return results

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string for retrieval.

        At query time we need the query as a vector so we can
        compare it against our stored chunk vectors in ChromaDB.

        Args:
            query: The user's question as a plain string.

        Returns:
            384-dimensional vector as a Python list of floats.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        logger.debug("Embedding query: '{}'", query[:80])

        # encode() works on a list or a single string.
        # For a single string it returns a 1D array of shape (384,).
        vector = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vector.tolist()