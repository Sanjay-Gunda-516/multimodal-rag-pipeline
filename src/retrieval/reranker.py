"""
Cross-Encoder Reranker — precision re-scoring of candidate chunks.

Model: BAAI/bge-reranker-base
    - Downloads ~278 MB on first use, cached at ~/.cache/huggingface/
    - BERT-base architecture (12 layers, 110M parameters)
    - State-of-the-art open-source reranker on BEIR benchmarks (2024-2025)
    - MPS compatible on Apple Silicon M1/M2/M3

Why rerank after RRF?
    BM25 and vector search both use bi-encoders:
        query  → embed → vector_q
        chunk  → embed → vector_d
        score  = cosine(vector_q, vector_d)
    Fast, but the query and document never interact — the model can't
    understand "is this chunk actually answering this question?"

    A cross-encoder reads BOTH together:
        score = model([query, chunk])
    Full attention between every query token and every chunk token.
    Much slower (can't pre-compute), but meaningfully more accurate.

    Strategy: RRF narrows 40 candidates to 20.
              Cross-encoder re-scores the top 20.
              We keep the best 5 for the LLM.
              Cost: 20 forward passes per query (fast on MPS).

MPS note:
    sentence-transformers CrossEncoder respects the device passed in.
    On M3, this runs on the Metal GPU for ~3-5× speedup over CPU.
"""

from typing import Dict, List

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from config import get_device, get_settings


class Reranker:
    """
    Re-scores a list of candidate chunks using a cross-encoder model.

    Usage:
        reranker = Reranker()                      # load model once
        results  = reranker.rerank(query, chunks)  # re-score top-20
        top_5    = results[:5]                     # take the best

    The reranker is always the last step before the LLM — it produces
    the final top-k that goes into the context window.
    """

    def __init__(self):
        self.settings = get_settings()
        self.device   = get_device()

        logger.info(
            "Loading reranker model '{}' on device='{}'",
            self.settings.reranker_model,
            self.device,
        )

        # CrossEncoder loads a BERT-style model fine-tuned to output
        # a single relevance score for (query, document) pairs.
        # max_length=512 covers most chunk sizes — truncates longer ones.
        self.model = CrossEncoder(
            self.settings.reranker_model,
            max_length=512,
            device=self.device,
        )

        logger.success(
            "Reranker ready | model='{}' | device='{}'",
            self.settings.reranker_model,
            self.device,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Re-score candidate chunks and return the top-k most relevant.

        Args:
            query:      The user's question as a plain string.
            candidates: Output from rrf_fusion() — list of result dicts.
                        Each dict must have a "document" key with chunk text.
            top_k:      How many chunks to return after reranking.
                        Default 5 — these go directly into the LLM prompt.

        Returns:
            Top-k result dicts, re-sorted by cross-encoder score.
            Each dict gets two new fields:
                "rerank_score": float  ← raw logit from cross-encoder
                "final_rank":   int    ← 1-indexed rank after reranking
        """
        if not candidates:
            logger.warning("rerank() called with empty candidate list")
            return []

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Clamp top_k to number of candidates
        top_k = min(top_k, len(candidates))

        logger.info(
            "Reranking {} candidates → top {} | query='{}'",
            len(candidates),
            top_k,
            query[:80],
        )

        # Build (query, document) pairs for batch inference.
        # CrossEncoder.predict() handles tokenisation and batching internally.
        pairs = [(query, c["document"]) for c in candidates]

        # predict() returns a numpy array of shape (n_candidates,)
        # Each value is the raw logit score — higher = more relevant.
        # No sigmoid is needed — we only care about relative ordering.
        # show_progress_bar=False keeps logs clean for production.
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        # Attach cross-encoder scores to candidate dicts
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            entry = candidate.copy()
            entry["rerank_score"] = float(score)
            scored_candidates.append(entry)

        # Sort by reranker score — highest first
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Add final_rank field (1-indexed)
        top_results = []
        for rank, candidate in enumerate(scored_candidates[:top_k], start=1):
            candidate["final_rank"] = rank
            top_results.append(candidate)

        logger.success(
            "Reranking complete | top-{} selected | "
            "best={:.4f} | worst={:.4f}",
            len(top_results),
            top_results[0]["rerank_score"],
            top_results[-1]["rerank_score"],
        )

        return top_results