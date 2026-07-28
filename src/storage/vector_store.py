"""
Vector Store — persists and retrieves embeddings using ChromaDB.

ChromaDB is a local vector database that stores:
    - Embedding vectors (for similarity search)
    - Document text    (returned alongside results)
    - Metadata dicts   (for filtering and citation display)

All data is persisted to disk at CHROMA_DIR so embeddings
survive restarts — we only embed a PDF once.

Two collections:
    text_chunks        → 384-dim MiniLM vectors over text chunks
    image_embeddings   → 512-dim CLIP vectors over extracted images

Why separate collections?
    Different vector dimensions cannot share a collection.
    MiniLM = 384-dim, CLIP = 512-dim — they must be separate.
    At retrieval time we query both and merge results via RRF (Stage 5).
"""

from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import CHROMA_DIR, get_settings
from src.embeddings import EmbeddedChunk, EmbeddedImage


class VectorStore:
    """
    Wrapper around ChromaDB with two persistent collections.

    All write operations are idempotent — adding the same chunk
    twice updates it rather than creating a duplicate. This means
    re-processing the same PDF is safe.

    Usage:
        store = VectorStore()
        store.add_chunks(embedded_chunks)
        store.add_images(embedded_images)

        # At query time:
        text_results  = store.query_text(query_vector, top_k=20)
        image_results = store.query_images(image_query_vector, top_k=2)
    """

    def __init__(self):
        self.settings = get_settings()

        # PersistentClient writes all data to CHROMA_DIR on disk.
        # Every add/update is immediately persisted — no explicit save needed.
        # anonymized_telemetry=False  disables ChromaDB phoning home.
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # get_or_create_collection:
        #   - If the collection exists on disk → loads it
        #   - If it doesn't exist yet → creates it
        # This makes __init__ safe to call on both first run and restarts.
        #
        # metadata={"hnsw:space": "cosine"} tells ChromaDB to use cosine
        # similarity for nearest-neighbour search. This matches our
        # L2-normalised vectors — cosine similarity = dot product for
        # unit-norm vectors.
        self.text_collection = self.client.get_or_create_collection(
            name=self.settings.text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.image_collection = self.client.get_or_create_collection(
            name=self.settings.image_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "VectorStore ready | text_collection='{}' ({} docs) | "
            "image_collection='{}' ({} docs) | path='{}'",
            self.settings.text_collection_name,
            self.text_collection.count(),
            self.settings.image_collection_name,
            self.image_collection.count(),
            CHROMA_DIR,
        )

    # ── Write Operations ───────────────────────────────────────────────────────

    def add_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """
        Store text chunk embeddings in ChromaDB.

        Uses upsert (update + insert) so re-processing the same PDF
        updates existing records instead of raising a duplicate key error.

        ChromaDB upsert requires four parallel lists of equal length:
            ids:        unique identifier per record
            embeddings: the vector for each record
            documents:  the raw text (returned in query results)
            metadatas:  dict of filterable/displayable fields
        """
        if not embedded_chunks:
            logger.warning("add_chunks() called with empty list")
            return

        logger.info(
            "Storing {} text chunks in ChromaDB...",
            len(embedded_chunks),
        )

        # ChromaDB expects four parallel lists — same index = same record
        ids        = [ec.chunk_id            for ec in embedded_chunks]
        embeddings = [ec.embedding           for ec in embedded_chunks]
        documents  = [ec.text                for ec in embedded_chunks]
        metadatas  = [ec.metadata            for ec in embedded_chunks]

        # upsert: insert new, update existing — idempotent
        self.text_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.success(
            "Stored {} text chunks | collection now has {} total docs",
            len(embedded_chunks),
            self.text_collection.count(),
        )

    def add_images(self, embedded_images: List[EmbeddedImage]) -> None:
        """
        Store image embeddings in ChromaDB.

        Same upsert pattern as add_chunks.
        The 'document' field stores the filename (images have no text).
        """
        if not embedded_images:
            logger.warning("add_images() called with empty list — skipping")
            return

        logger.info(
            "Storing {} image embeddings in ChromaDB...",
            len(embedded_images),
        )

        ids        = [ei.filename              for ei in embedded_images]
        embeddings = [ei.embedding             for ei in embedded_images]
        documents  = [ei.filename              for ei in embedded_images]
        metadatas  = [ei.to_metadata_dict()    for ei in embedded_images]

        self.image_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.success(
            "Stored {} images | collection now has {} total docs",
            len(embedded_images),
            self.image_collection.count(),
        )

    # ── Read Operations ────────────────────────────────────────────────────────

    def query_text(
        self,
        query_vector: List[float],
        top_k: int = 20,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Find the most similar text chunks to a query vector.

        Args:
            query_vector: 384-dim embedding from TextEmbedder.embed_query()
            top_k:        Number of results to return (default 20 for RRF)
            where:        Optional ChromaDB metadata filter e.g.
                          {"source_file": "paper.pdf"} to search one doc only

        Returns:
            List of dicts, each containing:
                {id, document, metadata, distance}
            Ordered by similarity (most similar first).
            distance here is cosine distance (0=identical, 2=opposite).
        """
        if self.text_collection.count() == 0:
            logger.warning("query_text() called on empty collection")
            return []

        # Clamp top_k to collection size to avoid ChromaDB errors
        top_k = min(top_k, self.text_collection.count())

        query_params = {
            "query_embeddings": [query_vector],
            "n_results":        top_k,
            # include: what to return in results
            # "distances"  → similarity scores
            # "documents"  → the raw text of each chunk
            # "metadatas"  → the metadata dict we stored
            "include": ["distances", "documents", "metadatas"],
        }
        if where:
            query_params["where"] = where

        raw = self.text_collection.query(**query_params)

        # ChromaDB returns results wrapped in extra lists (batch dimension).
        # raw["ids"] = [[id1, id2, ...]] — note the outer list.
        # We always query one vector at a time, so [0] unwraps it.
        return self._parse_results(raw)

    def query_images(
        self,
        query_vector: List[float],
        top_k: int = 2,
    ) -> List[Dict]:
        """
        Find the most visually/semantically relevant images to a query.

        Args:
            query_vector: 512-dim CLIP text embedding from
                          ImageEmbedder.embed_query_text()
            top_k:        How many images to return (default 2)

        Returns:
            Same structure as query_text results.
        """
        if self.image_collection.count() == 0:
            logger.warning("query_images() called on empty collection")
            return []

        top_k = min(top_k, self.image_collection.count())

        raw = self.image_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["distances", "documents", "metadatas"],
        )

        return self._parse_results(raw)

    # ── Utility ────────────────────────────────────────────────────────────────

    def _parse_results(self, raw: Dict) -> List[Dict]:
        """
        Flatten ChromaDB's batched result format into a clean list of dicts.

        ChromaDB returns:
            {"ids": [[id1, id2]], "distances": [[d1, d2]], ...}

        We return:
            [{"id": id1, "distance": d1, "document": ..., "metadata": ...}, ...]
        """
        ids       = raw["ids"][0]
        distances = raw["distances"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]

        return [
            {
                "id":       ids[i],
                "distance": distances[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }
            for i in range(len(ids))
        ]

    def get_text_count(self) -> int:
        """Total number of text chunks stored."""
        return self.text_collection.count()

    def get_image_count(self) -> int:
        """Total number of images stored."""
        return self.image_collection.count()

    def clear_all(self) -> None:
        """
        Delete all records from both collections.
        Used in tests and when reprocessing all documents from scratch.
        """
        self.client.delete_collection(self.settings.text_collection_name)
        self.client.delete_collection(self.settings.image_collection_name)

        # Recreate empty collections after deleting
        self.text_collection = self.client.get_or_create_collection(
            name=self.settings.text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.client.get_or_create_collection(
            name=self.settings.image_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("VectorStore cleared — all collections empty")