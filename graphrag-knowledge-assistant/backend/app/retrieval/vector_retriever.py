"""
Vector-based retrieval using ChromaDB + embeddings.

Takes a text query, embeds it, and finds the most similar
chunks in the vector store. Returns results with similarity
scores and source metadata.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.embeddings.embedding_service import embed_query
from app.vector.chroma_store import query_embeddings

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """A single vector search result with score and source info."""
    chunk_id: str
    content: str
    score: float  # cosine similarity (0–1, higher = more similar)
    document_id: str
    filename: str
    chunk_index: int


def vector_search(
    query: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> list[VectorSearchResult]:
    """Search for chunks similar to the query text.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results.
        document_id: If provided, restrict search to this document.

    Returns:
        List of VectorSearchResult sorted by similarity (highest first).
    """
    # Embed the query
    query_vector = embed_query(query)

    # Build optional filter
    where = {"document_id": document_id} if document_id else None

    # Query ChromaDB
    results = query_embeddings(
        query_embedding=query_vector,
        n_results=top_k,
        where=where,
    )

    # Parse results into structured objects
    search_results: list[VectorSearchResult] = []
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            metadata = results["metadatas"][0][i]
            search_results.append(VectorSearchResult(
                chunk_id=chunk_id,
                content=results["documents"][0][i],
                score=round(similarity, 4),
                document_id=metadata.get("document_id", ""),
                filename=metadata.get("filename", ""),
                chunk_index=metadata.get("chunk_index", 0),
            ))

    logger.info(
        "Vector search for '%s': %d results (top_k=%d)",
        query[:50], len(search_results), top_k,
    )
    return search_results
