# src/validation/hallucination_scorer.py
"""
Hallucination Scorer — verifies each claim in a generated answer against
source chunks using Natural Language Inference (NLI).

Model: cross-encoder/nli-deberta-v3-small
    - Downloads ~180 MB on first use
    - DeBERTa-v3 architecture, fine-tuned on NLI datasets
    - Outputs: CONTRADICTION / NEUTRAL / ENTAILMENT per (premise, hypothesis) pair
    - MPS compatible on Apple Silicon

Why NLI over LLM-as-judge?
    - Runs locally — zero additional API cost per query
    - Deterministic — same input always gives same score
    - Explainable — per-sentence labels show exactly which claims are suspect
    - Fast — DeBERTa-small does ~50 pairs/second on MPS

Scoring logic:
    For each sentence in the answer:
        1. Compare against every source chunk
        2. Take the maximum ENTAILMENT score across all chunks
           (a sentence only needs ONE source to support it)
        3. If max_entailment > threshold (0.5) → SUPPORTED
           Else → UNVERIFIED

    Trust score = SUPPORTED sentences / total sentences × 100%

Limitations (be ready to discuss in interviews):
    - Short sentences score lower (less context for NLI)
    - Technical jargon can confuse the NLI model
    - Paraphrased claims may score lower than direct quotes
    - Not a replacement for human evaluation on critical use cases
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from config import get_device, get_settings
from src.generation.llm_client import GeneratedAnswer


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class SentenceVerdict:
    """
    Verification result for a single sentence from the answer.

    Attributes:
        sentence:           The sentence being checked.
        max_entailment:     Highest entailment score across all source chunks.
        best_source_index:  Which source chunk gave the highest entailment.
        label:              SUPPORTED / UNVERIFIED / TOO_SHORT
        is_supported:       True if max_entailment > threshold.
    """
    sentence:           str
    max_entailment:     float
    best_source_index:  int
    label:              str
    is_supported:       bool


@dataclass
class HallucinationReport:
    """
    Complete hallucination analysis for one generated answer.

    Attributes:
        verdicts:       Per-sentence verification results.
        trust_score:    0.0–100.0 — percentage of supported sentences.
        supported:      Count of sentences that pass verification.
        unverified:     Count of sentences that fail verification.
        skipped:        Count of sentences too short to verify meaningfully.
        summary:        Human-readable one-line summary.
    """
    verdicts:    List[SentenceVerdict]
    trust_score: float
    supported:   int
    unverified:  int
    skipped:     int
    summary:     str = field(init=False)

    def __post_init__(self):
        if self.trust_score >= 80:
            grade = "HIGH"
        elif self.trust_score >= 50:
            grade = "MEDIUM"
        else:
            grade = "LOW"

        self.summary = (
            f"Trust: {self.trust_score:.1f}% ({grade}) | "
            f"{self.supported} supported | "
            f"{self.unverified} unverified | "
            f"{self.skipped} skipped"
        )


# ── Hallucination Scorer ───────────────────────────────────────────────────────

class HallucinationScorer:
    """
    Scores a GeneratedAnswer for groundedness using NLI.

    Usage:
        scorer  = HallucinationScorer()
        report  = scorer.score(generated_answer)
        print(report.trust_score)   # e.g. 87.5
        print(report.summary)       # "Trust: 87.5% (HIGH) | 7 supported..."
    """

    # Minimum character length to attempt NLI scoring.
    # Short sentences ("Yes.", "See above.") don't have enough
    # semantic content for reliable NLI classification.
    MIN_SENTENCE_LENGTH = 30

    # Entailment probability threshold.
    # If the best-matching source chunk has entailment probability
    # above this value, the sentence is considered supported.
    ENTAILMENT_THRESHOLD = 0.5

    def __init__(self):
        self.settings = get_settings()
        self.device   = get_device()

        logger.info(
            "Loading NLI model '{}' on device='{}'",
            self.settings.nli_model,
            self.device,
        )

        # CrossEncoder for NLI:
        # Input:  (premise, hypothesis) = (source_chunk, answer_sentence)
        # Output: [contradiction_score, neutral_score, entailment_score]
        # We read index [2] for entailment probability.
        self.model = CrossEncoder(
            self.settings.nli_model,
            device=self.device,
        )

        logger.success(
            "HallucinationScorer ready | model='{}' | threshold={} | device='{}'",
            self.settings.nli_model,
            self.entailment_threshold,
            self.device,
        )

    @property
    def entailment_threshold(self) -> float:
        return self.ENTAILMENT_THRESHOLD

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, generated_answer: GeneratedAnswer) -> HallucinationReport:
        """
        Score a generated answer for hallucination.

        Args:
            generated_answer: Output from LLMClient.generate()
                              Contains both the answer text and the
                              source chunks it was generated from.

        Returns:
            HallucinationReport with per-sentence verdicts and trust score.
        """
        answer_text  = generated_answer.answer
        source_texts = generated_answer.context.chunk_texts

        if not source_texts:
            logger.error("No source chunks to score against")
            raise ValueError("GeneratedAnswer has no source chunks in context")

        # Step 1: Split answer into sentences
        sentences = self._split_into_sentences(answer_text)

        logger.info(
            "Scoring {} sentences against {} source chunks",
            len(sentences),
            len(source_texts),
        )

        # Step 2: Score each sentence
        verdicts = []
        for sentence in sentences:
            verdict = self._verify_sentence(sentence, source_texts)
            verdicts.append(verdict)

        # Step 3: Compute trust score
        report = self._compute_report(verdicts)

        logger.success("Hallucination scoring complete | {}", report.summary)
        return report

    # ── Private: Sentence Splitting ────────────────────────────────────────────

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split answer text into individual sentences for per-claim checking.

        Handles:
        - Markdown headers (### Problem Solved)  → skip
        - Bullet points (- claim [Source 1])     → keep
        - Regular sentences                       → keep
        - Citation markers ([Source N])           → preserved in sentence
        """
        # Remove markdown headers — they're structural, not factual claims
        text = re.sub(r'^#{1,6}\s+.+$', '', text, flags=re.MULTILINE)

        # Split on sentence boundaries: period/exclamation/question
        # followed by space + capital, or newline
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)', text)

        sentences = []
        for s in raw_sentences:
            s = s.strip()
            # Remove leading bullet/dash markers
            s = re.sub(r'^[-•*]\s+', '', s)
            if s:
                sentences.append(s)

        return sentences

    # ── Private: NLI Scoring ───────────────────────────────────────────────────

    def _verify_sentence(
        self,
        sentence: str,
        source_texts: List[str],
    ) -> SentenceVerdict:
        """
        Check a single sentence against all source chunks using NLI.

        Returns the verdict with the highest entailment score found.
        """
        # Skip very short sentences — NLI is unreliable on them
        if len(sentence) < self.MIN_SENTENCE_LENGTH:
            return SentenceVerdict(
                sentence          = sentence,
                max_entailment    = 0.0,
                best_source_index = -1,
                label             = "TOO_SHORT",
                is_supported      = False,
            )

        # Build (source_chunk, answer_sentence) pairs for batch NLI
        # Note: NLI convention is (premise, hypothesis)
        #       premise   = the source text (what we know is true)
        #       hypothesis = the claim we want to verify
        pairs = [(source, sentence) for source in source_texts]

        # model.predict() returns shape (n_pairs, 3):
        # [:, 0] = contradiction scores
        # [:, 1] = neutral scores
        # [:, 2] = entailment scores
        # We use softmax to convert logits to probabilities
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            apply_softmax=True,   # convert logits → probabilities
        )

        # Extract entailment column (index 2)
        entailment_scores = [float(s[2]) for s in scores]

        # A sentence is supported if ANY source chunk entails it
        max_entailment    = max(entailment_scores)
        best_source_index = entailment_scores.index(max_entailment)
        is_supported      = max_entailment >= self.ENTAILMENT_THRESHOLD

        return SentenceVerdict(
            sentence          = sentence,
            max_entailment    = round(max_entailment, 4),
            best_source_index = best_source_index,
            label             = "SUPPORTED" if is_supported else "UNVERIFIED",
            is_supported      = is_supported,
        )

    # ── Private: Report ────────────────────────────────────────────────────────

    def _compute_report(
        self,
        verdicts: List[SentenceVerdict],
    ) -> HallucinationReport:
        """Aggregate per-sentence verdicts into a HallucinationReport."""
        supported  = sum(1 for v in verdicts if v.label == "SUPPORTED")
        unverified = sum(1 for v in verdicts if v.label == "UNVERIFIED")
        skipped    = sum(1 for v in verdicts if v.label == "TOO_SHORT")

        # Trust score uses only scoreable sentences (exclude TOO_SHORT)
        scoreable = supported + unverified
        if scoreable == 0:
            trust_score = 0.0
        else:
            trust_score = round((supported / scoreable) * 100, 1)

        return HallucinationReport(
            verdicts    = verdicts,
            trust_score = trust_score,
            supported   = supported,
            unverified  = unverified,
            skipped     = skipped,
        )