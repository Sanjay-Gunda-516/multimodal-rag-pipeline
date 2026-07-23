"""
Public interface for the document ingestion package.

Other parts of the project should import from this package instead of
importing individual ingestion modules directly.

Example:
    from src.ingestion import IngestionPipeline
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from loguru import logger

from .pdf_parser import ExtractedImage, PDFParser, PageText
from .text_chunker import TextChunk, TextChunker


@dataclass
class IngestionResult:
    """
    Complete output produced from ingesting one PDF.

    Attributes:
        pdf_name: Original PDF filename.
        chunks: Text chunks ready for embedding.
        images: Extracted images ready for image embedding.
    """

    pdf_name: str
    chunks: List[TextChunk]
    images: List[ExtractedImage]

    def summary(self) -> str:
        """Return a readable summary of the ingestion result."""
        return (
            f"'{self.pdf_name}' → "
            f"{len(self.chunks)} chunks | "
            f"{len(self.images)} images"
        )


class IngestionPipeline:
    """
    Coordinates PDF parsing and text chunking behind one simple interface.

    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.run(Path("paper.pdf"))
    """

    def __init__(self) -> None:
        self._parser = PDFParser()
        self._chunker = TextChunker()
        logger.info("IngestionPipeline ready")

    def run(self, pdf_path: Path) -> IngestionResult:
        """
        Parse and chunk one PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            IngestionResult containing text chunks and extracted images.
        """

        pdf_path = Path(pdf_path)
        logger.info("IngestionPipeline.run('{}')", pdf_path.name)

        page_texts, images = self._parser.parse(pdf_path)
        chunks = self._chunker.chunk(page_texts)

        return IngestionResult(
            pdf_name=pdf_path.name,
            chunks=chunks,
            images=images,
        )


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "TextChunk",
    "ExtractedImage",
    "PageText",
]