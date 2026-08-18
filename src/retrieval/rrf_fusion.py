"""
Reciprocal Rank Fusion — merges multiple ranked result lists into one.

RRF is the standard algorithm for combining heterogeneous search results.
It is used in production by Elasticsearch, Qdrant, Weaviate, and most
modern search systems because:
    - It is parameter-free (k=60 is a well-established default)
    - It handles different score scales gracefully (BM25 vs cosine)
    - It empirically outperforms linear score interpolation on BEIR benchmarks
    - It rewards documents that appear in MULTIPLE lists

Formula:
    RRF_score(d) = Σ  1 / (k + rank(d, R_i))
                  i
    where R_i is the i-th ranked list and rank() is 1-indexed.

Reference: Cormack, Clarke, Buettcher (2009) — "Reciprocal Rank Fusion
           outperforms Condorcet and individual Rank Learning Methods"
"""

from typing import Dict, List

from loguru import logger


# k=60 is the standard constant from the original RRF paper.
# Higher k → more weight to lower-ranked documents (flattens scores).
# Lower k  → top-ranked documents dominate even more.
# 60 is the empirically validated default used by Elasticsearch.
RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    top_k: int = 20,
) -> List[Dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of ranked result lists. Each inner list contains
                      dicts with at minimum: {"id": str, "document": str,
                      "metadata": dict}. The "score" field is ignored —
                      RRF uses only the rank position.
        top_k:        Maximum results to return after merging.

    Returns:
        A single merged and re-ranked list of result dicts, ordered by
        RRF score (highest first). Each dict contains:
            {
                "id":        str,
                "rrf_score": float,     ← the computed RRF score
                "document":  str,
                "metadata":  dict,
                "rank":      int,       ← 1-indexed rank in merged list
            }
    """
    if not result_lists:
        logger.warning("rrf_fusion called with empty result_lists")
        return []

    # ── Step 1: Compute RRF scores ─────────────────────────────────────────
    # scores maps: document_id → cumulative RRF score
    scores:    Dict[str, float] = {}

    # payload maps: document_id → the full result dict (for reconstruction)
    # We keep the first occurrence of each doc since all lists have same data.
    payloads:  Dict[str, Dict]  = {}

    for result_list in result_lists:
        if not result_list:
            continue  # skip empty lists (e.g. image collection is empty)

        for rank_0based, result in enumerate(result_list):
            doc_id = result["id"]
            # rank is 1-indexed in the RRF formula
            rank_1based = rank_0based + 1

            # Core RRF formula
            rrf_contribution = 1.0 / (RRF_K + rank_1based)

            # Accumulate — documents in multiple lists earn this multiple times
            scores[doc_id]   = scores.get(doc_id, 0.0) + rrf_contribution
            payloads[doc_id] = result  # store first-seen payload

    # ── Step 2: Sort by RRF score ──────────────────────────────────────────
    sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

    # ── Step 3: Build output list ──────────────────────────────────────────
    fused = []
    for rank, doc_id in enumerate(sorted_ids[:top_k], start=1):
        payload = payloads[doc_id].copy()
        payload["rrf_score"] = round(scores[doc_id], 6)
        payload["rank"]      = rank
        # Remove original score — it's from a different scale and misleading
        payload.pop("score", None)
        fused.append(payload)

    # ── Log summary ────────────────────────────────────────────────────────
    input_total  = sum(len(r) for r in result_lists)
    unique_total = len(scores)
    logger.debug(
        "RRF fusion: {} input results → {} unique → top-{} output | "
        "best_rrf={:.5f} worst_rrf={:.5f}",
        input_total,
        unique_total,
        len(fused),
        fused[0]["rrf_score"] if fused else 0,
        fused[-1]["rrf_score"] if fused else 0,
    )

    return fused