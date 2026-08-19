# src/generation/context_builder.py
"""
Context Builder — assembles retrieval results into a structured LLM prompt.

This is the most important prompt engineering component in the pipeline.
The format we choose directly affects:
    - Whether Claude cites sources correctly
    - Whether Claude stays grounded (low hallucination)
    - Whether Claude can answer multi-part questions
    - Token efficiency (shorter prompt = faster + cheaper)

Design decisions:
    1. Numbered sources — [Source N] labels make citation unambiguous.
       "As shown in [Source 3]..." is clearer than "According to page 5..."

    2. Metadata in every source header — file name and page number give
       Claude the information it needs to write citations without guessing.

    3. Images described inline — we can't send images to a text-only
       context window, but we tell Claude images exist and what page they're
       from. In Stage 8 (Streamlit UI) we'll display them separately.

    4. Strict instruction block — explicit "use ONLY the sources" instruction
       measurably reduces hallucination. This is the foundation our NLI
       scorer (Stage 7) verifies against.

    5. Low temperature (0.1 in config) — set at config level, not here.
       Temperature affects the LLM call, not the prompt construction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from config import get_settings


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class BuiltContext:
    """
    The assembled prompt and associated metadata.

    Attributes:
        system_prompt:   The system message defining Claude's role.
        user_prompt:     The full user message with sources + question.
        source_map:      Maps "[Source N]" → metadata dict for citation display.
        image_map:       Maps "[Image N]"  → image metadata for UI display.
        chunk_texts:     Raw chunk texts for hallucination scoring (Stage 7).
        total_tokens_est: Rough token estimate (chars / 4) for cost awareness.
    """
    system_prompt:    str
    user_prompt:      str
    source_map:       Dict[str, Dict]
    image_map:        Dict[str, Dict]
    chunk_texts:      List[str]
    total_tokens_est: int


# ── Context Builder ────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Assembles retrieval results into a structured, citation-ready LLM prompt.

    Usage:
        builder = ContextBuilder()
        context = builder.build(
            query        = "How does attention work?",
            text_results = final_top5,    # from reranker
            image_results= image_results, # from vector image search
        )
        # context.system_prompt → str
        # context.user_prompt   → str
        # context.source_map    → {"[Source 1]": {metadata...}, ...}
    """

    # System prompt defines Claude's role and core behaviour.
    # Kept short — detailed instructions go in the user prompt
    # where Claude can reference them alongside the actual sources.
    SYSTEM_PROMPT = """You are a precise document assistant. Your job is to \
answer questions using ONLY the provided source documents. You must cite every \
factual claim with its source number. Never use knowledge from outside the \
provided sources. If the sources do not contain enough information to answer \
the question, say so explicitly."""

    def __init__(self):
        self.settings = get_settings()
        logger.info("ContextBuilder initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        query:         str,
        text_results:  List[Dict],
        image_results: Optional[List[Dict]] = None,
    ) -> BuiltContext:
        """
        Build a complete prompt from retrieval results and a user query.

        Args:
            query:         The user's question as a plain string.
            text_results:  Top-k text chunks from Reranker.rerank()
            image_results: Top-k image results from VectorRetriever.search_images()

        Returns:
            BuiltContext with system_prompt, user_prompt, and metadata maps.
        """
        if not text_results:
            raise ValueError("Cannot build context with zero text results")

        image_results = image_results or []

        logger.info(
            "Building context | {} text chunks | {} images | query='{}'",
            len(text_results),
            len(image_results),
            query[:80],
        )

        # Build numbered source blocks
        text_block, source_map, chunk_texts = self._build_text_block(text_results)
        image_block, image_map              = self._build_image_block(image_results)

        # Assemble the full user prompt
        user_prompt = self._assemble_prompt(
            query=query,
            text_block=text_block,
            image_block=image_block,
            has_images=bool(image_results),
        )

        total_tokens_est = len(user_prompt) // 4

        logger.success(
            "Context built | {} sources | {} images | ~{} tokens",
            len(text_results),
            len(image_results),
            total_tokens_est,
        )

        return BuiltContext(
            system_prompt    = self.SYSTEM_PROMPT,
            user_prompt      = user_prompt,
            source_map       = source_map,
            image_map        = image_map,
            chunk_texts      = chunk_texts,
            total_tokens_est = total_tokens_est,
        )

    # ── Private: Text Block ────────────────────────────────────────────────────

    def _build_text_block(
        self,
        text_results: List[Dict],
    ) -> tuple:
        """
        Format text chunks into a numbered source block.

        Returns:
            (text_block_str, source_map, chunk_texts)
        """
        lines      = ["RETRIEVED TEXT SOURCES:", ""]
        source_map = {}
        chunk_texts = []

        for i, result in enumerate(text_results, start=1):
            source_key = f"[Source {i}]"
            metadata   = result.get("metadata", {})
            text       = result.get("document", "")

            source_file  = metadata.get("source_file",  "unknown")
            page_number  = metadata.get("page_number",  "?")

            # Source header — gives Claude the citation information
            header = f"{source_key} | {source_file} | Page {page_number}"
            lines.append(header)
            lines.append("-" * len(header))
            lines.append(text.strip())
            lines.append("")  # blank line between sources

            # Store metadata for UI citation display
            source_map[source_key] = {
                "source_file":  source_file,
                "page_number":  page_number,
                "chunk_index":  metadata.get("chunk_index", 0),
                "word_count":   metadata.get("word_count", 0),
                "rerank_score": result.get("rerank_score", 0.0),
                "text_preview": text[:200],
            }
            chunk_texts.append(text)

        return "\n".join(lines), source_map, chunk_texts

    # ── Private: Image Block ───────────────────────────────────────────────────

    def _build_image_block(
        self,
        image_results: List[Dict],
    ) -> tuple:
        """
        Format image results into a numbered image reference block.

        We can't embed images in a text-only prompt, so we describe them.
        The Streamlit UI (Stage 8) will display them visually alongside
        the text answer.
        """
        if not image_results:
            return "", {}

        lines     = ["RETRIEVED IMAGES:", ""]
        image_map = {}

        for i, result in enumerate(image_results, start=1):
            image_key = f"[Image {i}]"
            metadata  = result.get("metadata", {})

            filename    = metadata.get("filename",    "unknown")
            source_file = metadata.get("source_file", "unknown")
            page_number = metadata.get("page_number", "?")
            width       = metadata.get("width",       "?")
            height      = metadata.get("height",      "?")
            file_path   = metadata.get("file_path",   "")

            lines.append(
                f"{image_key} | {source_file} | Page {page_number} | "
                f"{width}×{height}px"
            )
            lines.append(
                f"Description: Image extracted from page {page_number} of "
                f"{source_file}. Dimensions: {width}×{height} pixels."
            )
            lines.append("")

            image_map[image_key] = {
                "filename":    filename,
                "source_file": source_file,
                "page_number": page_number,
                "width":       width,
                "height":      height,
                "file_path":   file_path,
            }

        return "\n".join(lines), image_map

    # ── Private: Assemble ──────────────────────────────────────────────────────

    def _assemble_prompt(
        self,
        query:      str,
        text_block: str,
        image_block: str,
        has_images: bool,
    ) -> str:
        """
        Combine all blocks into the final user prompt.
        """
        sections = [text_block]

        if has_images and image_block:
            sections.append(image_block)

        sections.append(f"USER QUESTION:\n{query}")

        # Instruction block — explicit rules reduce hallucination
        image_instruction = (
            " Reference images as [Image N] when relevant."
            if has_images else ""
        )

        instructions = f"""INSTRUCTIONS:
1. Answer the question using ONLY the sources listed above.
2. Cite every factual claim using [Source N] notation inline.{image_instruction}
3. If different sources say different things, note the discrepancy.
4. If the sources do not contain enough information, say: \
"The provided documents do not contain sufficient information to answer this."
5. Be concise and precise. Do not add information not in the sources."""

        sections.append(instructions)

        return "\n\n".join(sections)