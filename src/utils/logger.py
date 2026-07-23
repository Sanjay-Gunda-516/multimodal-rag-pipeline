# src/utils/logger.py
"""
Centralised logging configuration using loguru.

How to use in any module across the project:

    from loguru import logger
    logger.info("Processing page 3 of 10")
    logger.warning("Image too small — skipping")
    logger.error("ChromaDB connection failed")

Call setup_logger() exactly once at application startup.
Every other module just imports logger directly from loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_dir: Path, log_level: str = "INFO") -> None:
    """
    Configure loguru with two sinks:
      1. Console — coloured output, INFO level and above
      2. File    — rotating daily file, DEBUG level and above

    Args:
        log_dir:   Directory where .log files will be written.
        log_level: Minimum level shown in console. File captures everything.

    Log levels in order of severity:
        DEBUG → INFO → SUCCESS → WARNING → ERROR → CRITICAL
    """

    # Remove loguru's default handler first.
    # Without this, you get duplicate log lines — one from the default
    # handler and one from ours.
    logger.remove()

    # ── Sink 1: Console ──────────────────────────────────────────────────────
    # Shows INFO and above in the terminal with colour.
    # {time}    → timestamp
    # {level}   → log level, padded to 8 chars for alignment
    # {name}    → module name (e.g. src.ingestion.pdf_parser)
    # {function}→ function name where logger was called
    # {line}    → line number — makes bugs trivial to find
    # {message} → the actual log message
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # ── Sink 2: Rotating file ────────────────────────────────────────────────
    # Captures DEBUG and above — more verbose than console.
    # rotation="10 MB"   → starts a new file when current hits 10 MB
    # retention="7 days" → deletes files older than 7 days automatically
    # compression="zip"  → compresses rotated files to save disk space
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
    )

    logger.info(
        "Logger ready | console_level={} | log_dir={}",
        log_level,
        log_dir,
    )