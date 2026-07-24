"""
Public interface for the embeddings package.

Other modules should import embedding components from this package instead
of importing individual files directly.
"""

from .image_embedder import EmbeddedImage, ImageEmbedder
from .text_embedder import EmbeddedChunk, TextEmbedder

__all__ = [
    "TextEmbedder",
    "ImageEmbedder",
    "EmbeddedChunk",
    "EmbeddedImage",
]