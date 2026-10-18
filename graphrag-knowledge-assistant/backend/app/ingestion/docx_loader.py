"""
DOCX text extraction using python-docx.

Extracts text from all paragraphs in document order.
Tables and headers are included as plain text.
"""

import logging
from pathlib import Path

from docx import Document

logger = logging.getLogger(__name__)


def extract_text_from_docx(file_path: str | Path) -> str:
    """Extract all text from a DOCX file.

    Paragraphs are joined by newlines. Empty paragraphs are
    preserved as blank lines to maintain structural cues.
    Raises ValueError if no text is found.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    doc = Document(str(file_path))

    paragraphs = [p.text for p in doc.paragraphs]

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                paragraphs.append(row_text)

    full_text = "\n".join(paragraphs).strip()

    if not full_text:
        raise ValueError(f"No text could be extracted from {file_path.name}")

    logger.info(
        "Extracted %d characters from %s",
        len(full_text), file_path.name,
    )
    return full_text
