# src/ingestion/pdf_parser.py
"""
PDF Parser — extracts text and images from PDF documents.

Two-library strategy:
    pdfplumber → text extraction (best layout and reading-order handling)
    PyMuPDF    → image extraction (fastest, gives raw bytes + dimensions)

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF — imported as fitz (its historical name)
import pdfplumber
from loguru import logger

from config import IMAGES_DIR, get_settings


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class PageText:
    """
    Represents the extracted text from a single PDF page.

    Attributes:
        page_number:  1-indexed page number (humans count from 1, not 0).
        text:         Raw extracted text. May be empty for image-only pages.
        source_file:  Original PDF filename (not full path — used in citations).
        word_count:   Computed automatically after __init__.
    """
    page_number: int
    text: str
    source_file: str
    word_count: int = field(init=False)  # init=False means we compute it ourselves

    def __post_init__(self):
        # __post_init__ runs automatically after the dataclass __init__.
        # We use it to compute derived fields.
        self.word_count = len(self.text.split()) if self.text else 0


@dataclass
class ExtractedImage:
    """
    Represents an image extracted from a PDF page and saved to disk.

    Attributes:
        file_path:    Absolute path to the saved image file.
        page_number:  Which page this image came from (1-indexed).
        image_index:  Which image on that page (0-indexed — can be multiple).
        source_file:  Original PDF filename.
        width:        Image width in pixels.
        height:       Image height in pixels.
    """
    file_path: Path
    page_number: int
    image_index: int
    source_file: str
    width: int
    height: int

    @property
    def filename(self) -> str:
        """Just the filename, without the full path."""
        return self.file_path.name


# ── PDF Parser Class ───────────────────────────────────────────────────────────

class PDFParser:
    """
    Parses PDF documents into structured text and image data.

    Usage:
        parser = PDFParser()
        page_texts, images = parser.parse(Path("research_paper.pdf"))

    The parse() method is the only public interface.
    Internal methods are prefixed with _ and should not be called directly.
    """

    # Images smaller than these dimensions are almost always decorative:
    # bullets, icons, borders, watermarks. We skip them.
    # Tune these values if you find real content images being skipped.
    MIN_IMAGE_WIDTH  = 100  # pixels
    MIN_IMAGE_HEIGHT = 100  # pixels

    def __init__(self):
        self.settings = get_settings()
        logger.info("PDFParser initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse(
        self, pdf_path: Path
    ) -> Tuple[List[PageText], List[ExtractedImage]]:
        """
        Parse a PDF file into structured text and image data.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            Tuple of:
                - List[PageText]:       one entry per page
                - List[ExtractedImage]: all content images from the whole PDF

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError:        If the path is not a .pdf file.
        """
        # Validate the input before doing any work.
        # Fail fast with a clear message — better than a confusing fitz error later.
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a .pdf file, got: {pdf_path.suffix}"
            )

        logger.info("Starting parse: '{}'", pdf_path.name)

        page_texts = self._extract_text(pdf_path)
        images     = self._extract_images(pdf_path)

        # Log a human-readable summary — useful when processing many PDFs
        non_empty_pages = sum(1 for p in page_texts if p.text.strip())
        total_words     = sum(p.word_count for p in page_texts)

        logger.success(
            "Parsed '{}' | {} pages ({} with text) | {} words | {} images",
            pdf_path.name,
            len(page_texts),
            non_empty_pages,
            total_words,
            len(images),
        )

        return page_texts, images

    # ── Private: Text Extraction ───────────────────────────────────────────────

    def _extract_text(self, pdf_path: Path) -> List[PageText]:
        """
        Extract text from every page using pdfplumber.

        pdfplumber opens the PDF and gives us a list of Page objects.
        Each Page has an extract_text() method that handles layout
        analysis internally — it figures out reading order and returns
        a single clean string.
        """
        page_texts: List[PageText] = []
        source_file = pdf_path.name

        try:
            # pdfplumber.open() is a context manager — automatically closes
            # the file when the with-block exits, even if an error occurs.
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.debug("Extracting text from {} pages", total_pages)

                for i, page in enumerate(pdf.pages):
                    page_number = i + 1  # convert 0-indexed to 1-indexed

                    # extract_text() returns a string or None.
                    # None happens on scanned/image-only pages with no text layer.
                    raw_text = page.extract_text()

                    # Normalise to always be a string, never None.
                    text = raw_text.strip() if raw_text else ""

                    page_texts.append(PageText(
                        page_number=page_number,
                        text=text,
                        source_file=source_file,
                    ))

                    if text:
                        logger.debug(
                            "Page {}/{} — {} words",
                            page_number, total_pages, len(text.split())
                        )
                    else:
                        # Log a warning — the page may be scanned.
                        # We still keep it in the list so page numbers stay aligned.
                        logger.warning(
                            "Page {}/{} — no text found "
                            "(scanned image page or empty page)",
                            page_number, total_pages,
                        )

        except Exception as e:
            logger.error(
                "Text extraction failed for '{}': {}", pdf_path.name, e
            )
            raise  # re-raise so the caller knows something went wrong

        return page_texts

    # ── Private: Image Extraction ──────────────────────────────────────────────

    def _extract_images(self, pdf_path: Path) -> List[ExtractedImage]:
        """
        Extract embedded images from every page using PyMuPDF (fitz).

        PDFs store images internally with an xref (cross-reference) number.
        fitz lets us iterate page images, get their xref, then call
        extract_image(xref) to get the raw bytes and file extension.
        """
        extracted: List[ExtractedImage] = []
        source_file = pdf_path.name

        # Each PDF gets its own subdirectory under data/images/
        # so images from different documents never collide.
        # pdf_path.stem = filename without extension, e.g. "research_paper"
        pdf_image_dir = IMAGES_DIR / pdf_path.stem
        pdf_image_dir.mkdir(parents=True, exist_ok=True)

        try:
            # fitz.open() returns a Document object
            doc = fitz.open(str(pdf_path))

            for page_index in range(len(doc)):
                page_number = page_index + 1
                page = doc[page_index]

                # get_images(full=True) returns a list of tuples.
                # Each tuple describes one image embedded on this page:
                # (xref, smask, width, height, bpc, colorspace, alt_colorspace,
                #  name, filter, referencer)
                # We only need xref (index 0), width (2), and height (3).
                image_list = page.get_images(full=True)

                if not image_list:
                    continue  # no images on this page — move on

                for img_index, img_info in enumerate(image_list):
                    xref   = img_info[0]
                    width  = img_info[2]
                    height = img_info[3]

                    # Filter out small decorative images
                    if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
                        logger.debug(
                            "Skipping small image {}x{} on page {}",
                            width, height, page_number,
                        )
                        continue

                    try:
                        # extract_image(xref) returns a dict:
                        # {"image": bytes, "ext": "png"/"jpeg"/...,
                        #  "width": int, "height": int, ...}
                        base_image  = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext   = base_image["ext"]

                        # Deterministic filename — easy to trace back to source:
                        # research_paper__page003__img00.png
                        filename = (
                            f"{pdf_path.stem}"
                            f"__page{page_number:03d}"   # zero-padded to 3 digits
                            f"__img{img_index:02d}"      # zero-padded to 2 digits
                            f".{image_ext}"
                        )
                        image_path = pdf_image_dir / filename

                        # Write raw bytes to disk
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        extracted.append(ExtractedImage(
                            file_path=image_path,
                            page_number=page_number,
                            image_index=img_index,
                            source_file=source_file,
                            width=width,
                            height=height,
                        ))

                        logger.debug(
                            "Saved: {} ({}x{})",
                            filename, width, height,
                        )

                    except Exception as img_err:
                        # A single bad image should not kill the whole PDF.
                        # Log and continue to the next image.
                        logger.warning(
                            "Could not extract image {} on page {}: {}",
                            img_index, page_number, img_err,
                        )
                        continue

            doc.close()

        except Exception as e:
            logger.error(
                "Image extraction failed for '{}': {}", pdf_path.name, e
            )
            raise

        return extracted