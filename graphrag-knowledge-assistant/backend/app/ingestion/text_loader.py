"""
Plain-text file loader.

Handles .txt and .md files with automatic encoding detection.
Tries UTF-8 first, then falls back to latin-1 (which never fails).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_txt(file_path: str | Path) -> str:
    """Read all text from a plain-text file.

    Raises ValueError if the file is empty.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    # Try UTF-8 first, fall back to latin-1
    for encoding in ("utf-8", "latin-1"):
        try:
            text = file_path.read_text(encoding=encoding).strip()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {file_path.name}")

    if not text:
        raise ValueError(f"No text found in {file_path.name}")

    logger.info(
        "Extracted %d characters from %s",
        len(text), file_path.name,
    )
    return text
