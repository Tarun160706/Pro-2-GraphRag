"""
Search API endpoint.

POST /search — Perform vector similarity search across ingested documents.
"""

import logging

from fastapi import APIRouter

from app.retrieval.vector_retriever import vector_search
from app.vector.chroma_store import get_collection_stats
from app.schemas.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Search for relevant document chunks using vector similarity.

    Embeds the query, searches ChromaDB, and returns ranked results
    with similarity scores.
    """
    results = vector_search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                document_id=r.document_id,
                filename=r.filename,
                chunk_index=r.chunk_index,
            )
            for r in results
        ],
    )


@router.get("/search/stats")
async def search_stats() -> dict:
    """Return vector store statistics."""
    return get_collection_stats()
