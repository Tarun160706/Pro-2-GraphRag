"""
Text chunker with configurable size and overlap.

Splitting strategy:
  1. Try to split on paragraph boundaries (double newline).
  2. If a paragraph is still too large, split on sentence boundaries.
  3. If a sentence is still too large, split on word boundaries.

Overlap ensures that context isn't lost at chunk boundaries,
which is critical for retrieval quality.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sentence boundary: period/question/exclamation followed by whitespace
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Default values tuned for all-MiniLM-L6-v2 (max 256 tokens ≈ ~1000 chars)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200


@dataclass
class TextChunk:
    """A chunk of text with its position in the original document."""
    content: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split text into overlapping chunks.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of TextChunk objects with positional info.
    """
    if not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    raw_chunks = _split_with_overlap(text, chunk_size, chunk_overlap)

    # Build TextChunk objects with character offsets
    chunks: list[TextChunk] = []
    search_start = 0
    for idx, content in enumerate(raw_chunks):
        # Find where this chunk starts in the original text
        start = text.find(content[:50], search_start)
        if start == -1:
            start = search_start  # fallback
        end = start + len(content)

        chunks.append(TextChunk(
            content=content,
            chunk_index=idx,
            start_char=start,
            end_char=end,
        ))
        # Next search starts after accounting for overlap
        search_start = max(start + 1, end - chunk_overlap)

    logger.info(
        "Chunked %d characters into %d chunks (size=%d, overlap=%d)",
        len(text), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def _split_with_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Core splitting logic: paragraph → sentence → word fallback."""
    # Split into paragraphs first
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        # If adding this paragraph stays within the limit
        if current_length + para_len + 1 <= chunk_size:
            current_chunk.append(para)
            current_length += para_len + 1  # +1 for the newline join
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append("\n".join(current_chunk))

            # If the paragraph itself is too big, split it further
            if para_len > chunk_size:
                sub_chunks = _split_large_paragraph(para, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                current_chunk = []
                current_length = 0
            else:
                current_chunk = [para]
                current_length = para_len

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # Apply overlap between consecutive chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)

    return chunks


def _split_large_paragraph(
    paragraph: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split a large paragraph by sentences, then by words if needed."""
    sentences = _SENTENCE_SPLIT.split(paragraph)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sent_len = len(sentence)

        if current_length + sent_len + 1 <= chunk_size:
            current.append(sentence)
            current_length += sent_len + 1
        else:
            if current:
                chunks.append(" ".join(current))

            # If a single sentence exceeds chunk_size, split by words
            if sent_len > chunk_size:
                word_chunks = _split_by_words(sentence, chunk_size)
                chunks.extend(word_chunks)
                current = []
                current_length = 0
            else:
                current = [sentence]
                current_length = sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_by_words(text: str, chunk_size: int) -> list[str]:
    """Last-resort split on word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + 1 <= chunk_size:
            current.append(word)
            current_length += word_len + 1
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
            current_length = word_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to the current chunk."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Take the last `overlap` characters from the previous chunk
        overlap_text = prev[-overlap:].strip()
        # Avoid starting mid-word: trim to the first space
        space_idx = overlap_text.find(" ")
        if space_idx > 0:
            overlap_text = overlap_text[space_idx + 1:]
        result.append(overlap_text + " " + chunks[i])
    return result
