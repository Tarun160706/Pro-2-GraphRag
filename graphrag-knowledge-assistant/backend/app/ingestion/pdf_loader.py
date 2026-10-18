"""
PDF text extraction using PyMuPDF (fitz).

PyMuPDF is fast, has no system dependencies, and handles most
PDF layouts well. Each page's text is extracted separately so
we can track page-level source references later.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF file.

    Returns the full document text with page breaks marked by
    double newlines. Raises ValueError if the file yields no text.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    pages: list[str] = []
    doc = fitz.open(str(file_path))
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append(text)
    finally:
        doc.close()

    if not pages:
        raise ValueError(f"No text could be extracted from {file_path.name}")

    full_text = "\n\n".join(pages)
    logger.info(
        "Extracted %d characters from %d pages of %s",
        len(full_text), len(pages), file_path.name,
    )
    return full_text
