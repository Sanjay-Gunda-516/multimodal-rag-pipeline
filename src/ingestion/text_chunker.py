# src/ingestion/text_chunker.py
"""
Text Chunker — splits extracted page text into embedding-ready chunks.

Why chunking matters:
    Embedding models map a piece of text to a single vector.
    A full page (~500 words) embedded as one vector loses too much
    detail — the vector becomes a vague average of all topics on the page.
    Smaller, focused chunks produce sharper, more retrievable embeddings.

Why overlapping chunks:
    If we cut every 500 tokens with no overlap, a sentence that falls
    across a boundary is split — the first half is in chunk N, the
    second half is in chunk N+1. Neither chunk has the full sentence.
    50-token overlap ensures every sentence appears whole in at least
    one chunk.

Why RecursiveCharacterTextSplitter:
    It tries to split on paragraph breaks first (\n\n),
    then sentence ends (\n), then spaces, and only as a last resort
    splits mid-word. This preserves natural text boundaries as much
    as possible.
"""

from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import get_settings
from src.ingestion.pdf_parser import PageText


# ── Data Structure ─────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A single embedding-ready chunk of text with full provenance metadata.

    The metadata fields are what enable source citations in the final answer:
    "This claim comes from research_paper.pdf, page 4, chunk 2."

    Attributes:
        text:          The actual chunk content — goes into the embedder.
        source_file:   Original PDF filename.
        page_number:   Which page this chunk came from (1-indexed).
        chunk_index:   Position of this chunk within its page (0-indexed).
        chunk_id:      Unique identifier across the entire document.
                       Format: "filename__page003__chunk00"
        word_count:    Number of words in this chunk.
    """
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    chunk_id: str = field(init=False)
    word_count: int = field(init=False)

    def __post_init__(self):
        # Build a deterministic, human-readable ID.
        # Using the same zero-padding scheme as pdf_parser.py for consistency.
        stem = self.source_file.replace(".pdf", "")
        self.chunk_id = (
            f"{stem}"
            f"__page{self.page_number:03d}"
            f"__chunk{self.chunk_index:02d}"
        )
        self.word_count = len(self.text.split()) if self.text else 0

    def to_metadata_dict(self) -> dict:
        """
        Returns a flat dict of all metadata fields.
        ChromaDB stores this alongside the embedding vector
        so we can retrieve it at query time for citation generation.
        """
        return {
            "chunk_id":    self.chunk_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "word_count":  self.word_count,
        }


# ── Text Chunker Class ─────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits a list of PageText objects into a flat list of TextChunks.

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk(page_texts)
    """

    def __init__(self):
        self.settings = get_settings()

        # RecursiveCharacterTextSplitter splits text by trying these
        # separators in order: paragraph → sentence → word → character.
        # It stops at the first separator that keeps chunks below chunk_size.
        #
        # chunk_size:    maximum characters per chunk (not tokens — we use
        #                characters here because it's faster and close enough
        #                for our MiniLM model which handles ~512 tokens max)
        # chunk_overlap: characters shared between consecutive chunks
        # length_function: how to measure chunk size (len = character count)
        # is_separator_regex: treat separators as literal strings, not regex
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size * 4,  # ~4 chars per token
            chunk_overlap=self.settings.chunk_overlap * 4,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            "TextChunker initialised | chunk_size={} | overlap={}",
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(self, page_texts: List[PageText]) -> List[TextChunk]:
        """
        Split a list of PageText objects into a flat list of TextChunks.

        Args:
            page_texts: Output from PDFParser.parse() — one per page.

        Returns:
            Flat list of TextChunk objects, ordered page by page.
            Empty pages are skipped — they produce no chunks.
        """
        all_chunks: List[TextChunk] = []
        skipped_pages = 0

        for page in page_texts:

            # Skip blank or nearly-blank pages.
            # A page with fewer than 50 characters has almost no information —
            # it's likely a scanned image page or a section divider.
            if not page.text or len(page.text.strip()) < 50:
                skipped_pages += 1
                logger.debug(
                    "Skipping page {} — only {} chars",
                    page.page_number,
                    len(page.text.strip()) if page.text else 0,
                )
                continue

            # Split this page's text into a list of strings
            raw_chunks = self.splitter.split_text(page.text)

            # Convert each string into a TextChunk with full metadata
            for chunk_index, chunk_text in enumerate(raw_chunks):

                # Skip chunks that are just whitespace after splitting
                if not chunk_text.strip():
                    continue

                chunk = TextChunk(
                    text=chunk_text.strip(),
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
                all_chunks.append(chunk)

        logger.success(
            "Chunking complete | {} chunks from {} pages "
            "({} pages skipped)",
            len(all_chunks),
            len(page_texts) - skipped_pages,
            skipped_pages,
        )

        return all_chunks