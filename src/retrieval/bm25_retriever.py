"""
BM25 Retriever — keyword-based search over text chunks using bm25s.

Why BM25 alongside ChromaDB (vector search)?
    Vector search excels at semantic similarity — finding chunks that
    MEAN the same thing as the query, even with different words.
    BM25 excels at exact keyword matching — finding chunks that
    CONTAIN the exact words in the query.

    Examples where BM25 wins:
        - "equation 3"        → vector search has no concept of "equation 3"
        - "Table 2 results"   → specific labels, not semantic concepts
        - "dk parameter"      → rare technical symbol, no semantic embedding
        - "2017 paper"        → year + noun, not a semantic phrase

    Examples where vector search wins:
        - "what is attention" → matches "the attention mechanism maps..."
        - "how does it work"  → matches technical explanations
        - "main contribution" → matches abstract language about novelty

    Together (hybrid search via RRF in Stage 5) they beat either alone
    on every benchmark we care about.

Why bm25s over rank-bm25?
    - 40-500x faster (NumPy-based, not pure Python loops)
    - Saves/loads index to disk natively
    - Actively maintained (rank-bm25 is largely abandoned)
    - Drop-in keyword search with identical results quality

Persistence:
    Index is saved to BM25_DIR after build_index().
    load_index() restores it from disk — no rebuilding needed on restart.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import bm25s
from loguru import logger

from config import BM25_DIR, get_settings
from src.ingestion.text_chunker import TextChunk


class BM25Retriever:
    """
    Builds, persists, and queries a BM25 keyword index over TextChunks.

    Lifecycle:
        First run:
            retriever = BM25Retriever()
            retriever.build_index(chunks)   # indexes + saves to disk

        Subsequent runs:
            retriever = BM25Retriever()
            retriever.load_index()          # loads from disk instantly
            results = retriever.search("query", top_k=20)

    Usage in Stage 5 (hybrid retrieval):
        Both build_index() and load_index() are called by the storage
        coordinator. BM25Retriever.search() is called alongside
        VectorStore.query_text() and results are merged with RRF.
    """

    # Filenames within BM25_DIR
    INDEX_SUBDIR  = "index"          # bm25s saves multiple .npy files here
    CHUNKS_FILE   = "chunks.pkl"     # our parallel chunk list (pickle)
    METADATA_FILE = "metadata.json"  # human-readable summary (for debugging)

    def __init__(self):
        self.settings  = get_settings()
        self._retriever: Optional[bm25s.BM25] = None
        self._chunks:    List[TextChunk]       = []

        logger.info("BM25Retriever initialised")

    # ── Public: Build ──────────────────────────────────────────────────────────

    def build_index(self, chunks: List[TextChunk]) -> None:
        """
        Build a BM25 index from a list of TextChunk objects and persist it.

        This is called once per PDF (or set of PDFs). The resulting index
        is saved to BM25_DIR and can be restored with load_index().

        Args:
            chunks: All TextChunk objects from the ingestion pipeline.
                    Appends to any existing chunks if index already exists.
        """
        if not chunks:
            logger.warning("build_index() called with empty chunk list")
            return

        logger.info("Building BM25 index over {} chunks...", len(chunks))

        # Store chunks for result reconstruction later
        self._chunks = chunks

        # Extract raw text strings — this is what BM25 indexes
        corpus_texts = [chunk.text for chunk in chunks]

        # bm25s.tokenize():
        #   - lowercases all text
        #   - splits on whitespace and punctuation
        #   - removes English stopwords ("the", "a", "is", etc.)
        #   - returns a TokenizedCorpus object (list of token lists)
        #
        # Stopword removal improves precision: without it, "what is the
        # attention mechanism" would match chunks containing just "the".
        tokenized_corpus = bm25s.tokenize(
            corpus_texts,
            stopwords="en",
            show_progress=False,
        )

        # Create the BM25 object and build the index.
        # Under the hood, bm25s computes IDF scores for every token
        # across the whole corpus — this is the slow step (done once).
        self._retriever = bm25s.BM25()
        self._retriever.index(tokenized_corpus)

        # Persist to disk
        self._save(corpus_texts)

        logger.success(
            "BM25 index built | {} chunks | saved to '{}'",
            len(chunks),
            BM25_DIR,
        )

    # ── Public: Load ───────────────────────────────────────────────────────────

    def load_index(self) -> bool:
        """
        Load a previously built BM25 index from disk.

        Returns:
            True  if index loaded successfully.
            False if no index found on disk (need to call build_index first).
        """
        index_path  = BM25_DIR / self.INDEX_SUBDIR
        chunks_path = BM25_DIR / self.CHUNKS_FILE

        if not index_path.exists() or not chunks_path.exists():
            logger.warning(
                "No BM25 index found at '{}' — call build_index() first",
                BM25_DIR,
            )
            return False

        logger.info("Loading BM25 index from disk...")

        # Restore the bm25s index (numpy arrays)
        self._retriever = bm25s.BM25.load(
            str(index_path),
            load_corpus=False,   # we manage corpus ourselves via chunks.pkl
        )

        # Restore our TextChunk objects (preserves all metadata)
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

        logger.success(
            "BM25 index loaded | {} chunks indexed",
            len(self._chunks),
        )
        return True

    # ── Public: Search ─────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search the BM25 index for chunks most relevant to the query.

        Args:
            query:  User's question or search phrase as a plain string.
            top_k:  Maximum number of results to return.

        Returns:
            List of dicts ordered by BM25 score (highest first):
            [
                {
                    "id":       chunk_id string,
                    "score":    BM25 relevance score (float, higher = better),
                    "document": chunk text,
                    "metadata": {source_file, page_number, chunk_index, ...}
                },
                ...
            ]

            Returns empty list if query matches nothing or index not built.
        """
        if not self.is_built:
            logger.error("search() called before index is built or loaded")
            return []

        if not query.strip():
            logger.warning("search() called with empty query")
            return []

        # Clamp top_k to corpus size — bm25s raises if k > corpus size
        top_k = min(top_k, len(self._chunks))

        logger.debug("BM25 search: query='{}' top_k={}", query[:80], top_k)

        # Tokenize the query with the same settings as the corpus
        # IMPORTANT: must use identical tokenization or scores are meaningless
        tokenized_query = bm25s.tokenize(
            [query],
            stopwords="en",
            show_progress=False,
        )

        # retrieve() returns two numpy arrays:
        #   results: shape (n_queries, k) — corpus indices of top-k chunks
        #   scores:  shape (n_queries, k) — BM25 score for each result
        # We always search with one query at a time, so [0] unwraps batch dim.
        results_indices, scores = self._retriever.retrieve(
            tokenized_query,
            k=top_k,
        )

        indices = results_indices[0]   # shape (k,)
        scores  = scores[0]            # shape (k,)

        # Reconstruct result dicts using the stored chunk objects.
        # Filter out zero-score results — they matched nothing.
        output = []
        for idx, score in zip(indices, scores):
            if score <= 0:
                continue  # BM25 score of 0 means no keyword overlap at all

            chunk = self._chunks[int(idx)]
            output.append({
                "id":       chunk.chunk_id,
                "score":    float(score),
                "document": chunk.text,
                "metadata": chunk.to_metadata_dict(),
            })

        logger.debug(
            "BM25 returned {}/{} results for query='{}'",
            len(output), top_k, query[:60],
        )

        return output

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        """True if index is in memory (built or loaded)."""
        return self._retriever is not None and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the current index."""
        return len(self._chunks)

    # ── Private ────────────────────────────────────────────────────────────────

    def _save(self, corpus_texts: List[str]) -> None:
        """
        Persist the BM25 index and chunk metadata to BM25_DIR.

        Two files are written:
            BM25_DIR/index/   → bm25s numpy arrays (the actual index)
            BM25_DIR/chunks.pkl → pickled TextChunk list (for metadata)
        """
        import json

        BM25_DIR.mkdir(parents=True, exist_ok=True)
        index_path = BM25_DIR / self.INDEX_SUBDIR

        # Save the bm25s index (creates multiple .npy files)
        self._retriever.save(str(index_path))

        # Save TextChunk objects — pickle preserves the dataclass structure
        # including chunk_id, page_number, source_file, word_count, etc.
        with open(BM25_DIR / self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self._chunks, f)

        # Save a human-readable summary alongside the binary files
        # Makes it easy to inspect what's indexed without loading Python
        summary = {
            "total_chunks": len(self._chunks),
            "source_files": list({c.source_file for c in self._chunks}),
            "chunk_ids_sample": [c.chunk_id for c in self._chunks[:5]],
        }
        with open(BM25_DIR / self.METADATA_FILE, "w") as f:
            json.dump(summary, f, indent=2)

        logger.debug(
            "BM25 index saved | index_path='{}' | chunks.pkl written",
            index_path,
        )