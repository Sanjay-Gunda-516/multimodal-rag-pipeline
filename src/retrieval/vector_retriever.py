"""
Vector Retriever — semantic search interface over ChromaDB.

Wraps VectorStore and normalises output format to match BM25Retriever
so both feed cleanly into RRF fusion with the same dict structure.

Distance → Similarity conversion:
    ChromaDB uses cosine distance: 0 = identical, 2 = opposite.
    We convert to similarity:      1 = identical, 0 = opposite.
    Formula: similarity = 1 - (distance / 2)

Why a separate class from VectorStore?
    VectorStore is the storage layer  — it manages ChromaDB collections.
    VectorRetriever is the retrieval layer — it normalises scores and
    provides a unified interface for RRF fusion. Separation of concerns:
    if we swap ChromaDB for Qdrant later, only VectorStore changes.
"""

from typing import Dict, List, Optional

from loguru import logger

from config import get_settings
from src.storage.vector_store import VectorStore


class VectorRetriever:
    """
    Semantic search over text chunks and images stored in ChromaDB.

    Two public methods:
        search_text(query_vector)   → top-k semantically similar text chunks
        search_images(query_vector) → top-k semantically similar images

    Both return the same dict format as BM25Retriever.search():
        [{"id": ..., "score": 0.0–1.0, "document": ..., "metadata": ...}]

    Usage:
        retriever = VectorRetriever(store)
        results   = retriever.search_text(query_vector, top_k=20)
    """

    def __init__(self, store: VectorStore):
        """
        Args:
            store: An initialised VectorStore with indexed chunks and images.
                   VectorRetriever does not own the store — it is injected.
                   This is called Dependency Injection: the caller controls
                   the store's lifecycle, not this class.
        """
        self.store    = store
        self.settings = get_settings()
        logger.info("VectorRetriever initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def search_text(
        self,
        query_vector: List[float],
        top_k: int = 20,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Find the most semantically similar text chunks to a query vector.

        Args:
            query_vector: 384-dim embedding from TextEmbedder.embed_query()
            top_k:        Number of results to return (default 20 for RRF)
            where:        Optional ChromaDB filter e.g. {"source_file": "x.pdf"}

        Returns:
            List of result dicts ordered by similarity (highest first):
            [
                {
                    "id":       chunk_id,
                    "score":    float 0.0–1.0  (1.0 = identical),
                    "document": chunk text,
                    "metadata": {source_file, page_number, chunk_index, ...}
                }
            ]
        """
        if self.store.get_text_count() == 0:
            logger.warning("search_text() called on empty text collection")
            return []

        raw = self.store.query_text(query_vector, top_k=top_k, where=where)

        # Convert ChromaDB distance to similarity score
        results = self._normalise_distances(raw)

        logger.debug(
            "Vector text search: top={} | best_score={:.4f}",
            len(results),
            results[0]["score"] if results else 0.0,
        )

        return results

    def search_images(
        self,
        query_vector: List[float],
        top_k: int = 2,
    ) -> List[Dict]:
        """
        Find the most semantically relevant images to a CLIP text query.

        Args:
            query_vector: 512-dim CLIP text embedding from
                          ImageEmbedder.embed_query_text()
            top_k:        Number of images to return (default 2)

        Returns:
            Same dict format as search_text() with image metadata.
        """
        if self.store.get_image_count() == 0:
            logger.warning("search_images() called on empty image collection")
            return []

        raw = self.store.query_images(query_vector, top_k=top_k)
        results = self._normalise_distances(raw)

        logger.debug(
            "Vector image search: top={} | best_score={:.4f}",
            len(results),
            results[0]["score"] if results else 0.0,
        )

        return results

    # ── Private ────────────────────────────────────────────────────────────────

    def _normalise_distances(self, raw_results: List[Dict]) -> List[Dict]:
        """
        Convert ChromaDB cosine distances to similarity scores.

        ChromaDB distance ∈ [0, 2]:
            0.0 = vectors are identical (cosine similarity = 1.0)
            1.0 = vectors are orthogonal (cosine similarity = 0.0)
            2.0 = vectors are opposite (cosine similarity = -1.0)

        Our similarity score ∈ [0, 1]:
            1.0 = identical
            0.5 = orthogonal
            0.0 = opposite

        Formula: similarity = 1 - (distance / 2)
        """
        normalised = []
        for r in raw_results:
            distance   = r["distance"]
            similarity = 1.0 - (distance / 2.0)

            normalised.append({
                "id":       r["id"],
                "score":    round(similarity, 6),
                "document": r["document"],
                "metadata": r["metadata"],
            })

        # Already sorted by ChromaDB (most similar first)
        # but re-sort by our score just to be explicit
        normalised.sort(key=lambda x: x["score"], reverse=True)
        return normalised