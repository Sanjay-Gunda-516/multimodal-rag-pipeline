"""
Image Embedder — converts images and text queries into CLIP vectors.

Model: CLIP ViT-B/32 (openai pretrained weights) via open-clip-torch

Why CLIP?
    CLIP (Contrastive Language-Image Pretraining) was trained on 400M
    image-text pairs to map images and text into the SAME vector space.
    This means:
        - embed an image   → 512-dim vector
        - embed text query → 512-dim vector
        - compare them with cosine similarity → find relevant images

    A query "attention mechanism diagram" will match a transformer
    architecture figure even if that image has no text in it.

Why open-clip-torch instead of openai/clip?
    - Proper PyPI package — no GitHub install needed
    - Actively maintained (openai/clip is largely abandoned)
    - Same weights when pretrained='openai'
    - Better MPS compatibility on Apple Silicon

MPS on M3:
    The CLIP vision transformer runs on Metal Performance Shaders.
    We move the model to MPS with .to(device), run inference with
    torch.no_grad(), then move results back to CPU for ChromaDB.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import open_clip
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from config import get_device, get_settings
from src.ingestion import ExtractedImage


# ── Data Structure ─────────────────────────────────────────────────────────────

@dataclass
class EmbeddedImage:
    """
    An ExtractedImage paired with its CLIP embedding vector.

    Attributes:
        image:     Original ExtractedImage (path + page metadata).
        embedding: 512-dim float list — CLIP visual embedding.
                   Directly comparable with CLIP text embeddings.
    """
    image:     ExtractedImage
    embedding: List[float]

    @property
    def filename(self) -> str:
        return self.image.filename

    @property
    def file_path(self) -> Path:
        return self.image.file_path

    @property
    def page_number(self) -> int:
        return self.image.page_number

    @property
    def source_file(self) -> str:
        return self.image.source_file

    def to_metadata_dict(self) -> dict:
        """
        Flat metadata dict stored alongside the vector in ChromaDB.
        Retrieved at query time to display image citations in the UI.
        """
        return {
            "filename":    self.image.filename,
            "file_path":   str(self.image.file_path),
            "page_number": self.image.page_number,
            "source_file": self.image.source_file,
            "width":       self.image.width,
            "height":      self.image.height,
        }


# ── Image Embedder ─────────────────────────────────────────────────────────────

class ImageEmbedder:
    """
    Embeds images and text queries into CLIP's shared 512-dim vector space.

    Two public methods:
        embed(images)            → List[EmbeddedImage]  (vision encoder)
        embed_query_text(query)  → List[float]          (text encoder)

    Both outputs live in the same vector space — enabling text-to-image
    retrieval by cosine similarity.

    Usage:
        embedder = ImageEmbedder()                       # load CLIP once
        embedded = embedder.embed(images)                # embed images
        qvec     = embedder.embed_query_text("diagram")  # embed query
    """

    def __init__(self):
        self.settings = get_settings()
        self.device   = get_device()

        logger.info(
            "Loading CLIP model '{}' pretrained='{}' on device='{}'",
            self.settings.clip_model,
            self.settings.clip_pretrained,
            self.device,
        )

        # create_model_and_transforms returns three objects:
        #   model:      the CLIP neural network (vision + text encoders)
        #   _:          training augmentation transforms — we don't need these
        #   preprocess: inference preprocessing pipeline
        #               (resize → 224x224, normalise pixel values to
        #                the mean/std CLIP was trained on)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.settings.clip_model,
            pretrained=self.settings.clip_pretrained,
        )

        # Move all model weights to target device
        self.model = self.model.to(self.device)

        # eval() disables dropout and batch norm — only needed during training.
        # Always call before inference or you get inconsistent outputs.
        self.model.eval()

        # Tokenizer: converts a text string into a (1, 77) integer tensor.
        # 77 is CLIP's fixed context window length.
        self.tokenizer = open_clip.get_tokenizer(self.settings.clip_model)

        # CLIP ViT-B/32 always outputs 512-dimensional vectors
        self.embedding_dim = 512

        logger.success(
            "Image embedder ready | model='{}' | dim={} | device='{}'",
            self.settings.clip_model,
            self.embedding_dim,
            self.device,
        )

    # ── Public: Embed Images ───────────────────────────────────────────────────

    def embed(self, images: List[ExtractedImage]) -> List[EmbeddedImage]:
        """
        Embed extracted PDF images using CLIP's vision encoder.

        Each image goes through:
            PIL open → preprocess (resize+normalise) → CLIP vision encoder
            → L2 normalise → 512-dim vector

        Args:
            images: Output from PDFParser._extract_images()

        Returns:
            List[EmbeddedImage] — each image paired with its vector.
            Failed images are skipped and logged (not silently dropped).
        """
        if not images:
            logger.warning("embed() called with empty image list")
            return []

        logger.info("Embedding {} images with CLIP...", len(images))
        results: List[EmbeddedImage] = []

        # torch.no_grad() disables gradient tracking.
        # During inference we never do backprop — this saves memory
        # and makes the forward pass faster.
        with torch.no_grad():
            for extracted_img in images:
                try:
                    # Load image from disk.
                    # .convert("RGB") ensures consistent 3-channel format —
                    # some PDFs embed RGBA or grayscale images.
                    pil_image = Image.open(extracted_img.file_path).convert("RGB")

                    # preprocess(): resize to 224x224, normalise channels.
                    # Result shape: (3, 224, 224)
                    # .unsqueeze(0): add batch dim → (1, 3, 224, 224)
                    # .to(self.device): move tensor to MPS/CPU
                    image_tensor = (
                        self.preprocess(pil_image)
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    # Vision encoder forward pass → shape (1, 512)
                    image_features = self.model.encode_image(image_tensor)

                    # L2 normalise along last dim so magnitude = 1.0
                    # Required for cosine similarity = dot product
                    image_features = F.normalize(image_features, dim=-1)

                    # .squeeze(0): remove batch dim → shape (512,)
                    # .cpu(): move from MPS back to CPU (ChromaDB needs CPU)
                    # .tolist(): numpy/tensor → Python list of floats
                    embedding = image_features.squeeze(0).cpu().tolist()

                    results.append(EmbeddedImage(
                        image=extracted_img,
                        embedding=embedding,
                    ))

                    logger.debug(
                        "Embedded: {} ({}x{})",
                        extracted_img.filename,
                        extracted_img.width,
                        extracted_img.height,
                    )

                except Exception as e:
                    # Never let one bad image crash the whole batch.
                    # Log clearly and continue.
                    logger.warning(
                        "Skipping image '{}' — embed failed: {}",
                        extracted_img.filename, e,
                    )
                    continue

        logger.success(
            "Embedded {}/{} images | dim={} | device='{}'",
            len(results), len(images),
            self.embedding_dim, self.device,
        )

        return results

    # ── Public: Embed Query Text ───────────────────────────────────────────────

    def embed_query_text(self, query: str) -> List[float]:
        """
        Embed a text query using CLIP's TEXT encoder.

        Why this method exists:
            At query time, we want to find images relevant to the user's
            question. We embed the query with CLIP's text encoder and
            compare against stored image vectors (from the vision encoder).
            Both live in the same 512-dim space — so the comparison works.

        This is fundamentally different from embed_query() in TextEmbedder:
            TextEmbedder.embed_query()      → for finding text chunks
            ImageEmbedder.embed_query_text() → for finding images

        Args:
            query: User's question or search phrase.

        Returns:
            512-dimensional CLIP text embedding as Python list of floats.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty")

        logger.debug("Embedding query with CLIP text encoder: '{}'", query[:80])

        with torch.no_grad():
            # tokenizer returns shape (1, 77) — padded/truncated to 77 tokens
            tokens = self.tokenizer([query]).to(self.device)

            # Text encoder forward pass → shape (1, 512)
            text_features = self.model.encode_text(tokens)

            # Normalise for cosine similarity
            text_features = F.normalize(text_features, dim=-1)

            return text_features.squeeze(0).cpu().tolist()