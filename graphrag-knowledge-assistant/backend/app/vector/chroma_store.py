"""
ChromaDB vector store with persistent local storage.

Provides a clean interface for storing and querying document
chunk embeddings. Uses a single collection for all chunks,
with metadata to filter by document.

ChromaDB handles persistence automatically when given a path.
"""

import logging
from typing import Optional

import chromadb

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — one client shared across the app
_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def get_client() -> chromadb.ClientAPI:
    """Get or create the persistent ChromaDB client."""
    global _client
    if _client is None:
        logger.info("Initializing ChromaDB at %s", settings.chroma_dir)
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the document chunks collection.

    Uses cosine similarity which pairs well with normalized
    embeddings from the embedding service.
    """
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d vectors)",
            settings.chroma_collection,
            _collection.count(),
        )
    return _collection


def add_embeddings(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Add embeddings to the vector store.

    Args:
        ids: Unique IDs for each embedding (typically chunk IDs).
        embeddings: Pre-computed embedding vectors.
        documents: The original text for each embedding.
        metadatas: Metadata dicts (document_id, filename, chunk_index, etc.).
    """
    collection = get_collection()

    # ChromaDB has a batch size limit, process in batches of 5000
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info("Added %d embeddings to ChromaDB", len(ids))


def query_embeddings(
    query_embedding: list[float],
    n_results: int = 5,
    where: Optional[dict] = None,
) -> dict:
    """Query the vector store for similar chunks.

    Args:
        query_embedding: The query vector.
        n_results: Number of results to return.
        where: Optional metadata filter (e.g. {"document_id": "..."}).

    Returns:
        ChromaDB query result dict with ids, documents, metadatas, distances.
    """
    collection = get_collection()

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    # Guard against empty collection
    if collection.count() == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    results = collection.query(**kwargs)
    return results


def delete_by_document(document_id: str) -> None:
    """Remove all embeddings belonging to a document.

    Called when a document is deleted to keep the vector store
    in sync with the SQLite database.
    """
    collection = get_collection()
    # ChromaDB where filter to find all chunks for this document
    existing = collection.get(where={"document_id": document_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info(
            "Deleted %d embeddings for document %s",
            len(existing["ids"]), document_id,
        )


def get_collection_stats() -> dict:
    """Return basic stats about the vector store."""
    collection = get_collection()
    return {
        "collection_name": settings.chroma_collection,
        "total_vectors": collection.count(),
        "chroma_dir": settings.chroma_dir,
    }
