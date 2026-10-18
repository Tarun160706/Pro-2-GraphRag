"""
Query & Answer Generation API endpoints for GraphRAG Knowledge Assistant.

POST /query — Generate grounded answer with source citations using Hybrid, Vector, or Graph retrieval.
POST /chat  — Alias for chat UI interaction.
"""

import logging

from fastapi import APIRouter

from app.llm.generator import generate_answer
from app.retrieval.hybrid_retriever import hybrid_search
from app.schemas.schemas import (
    GraphEdgeItem,
    GraphNodeItem,
    QueryRequest,
    QueryResponse,
    SourceCitation,
    SubgraphData,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["GraphRAG Intelligence"])


@router.post("/query", response_model=QueryResponse)
async def query_assistant(req: QueryRequest) -> QueryResponse:
    """Execute GraphRAG retrieval (Hybrid, Vector, or Graph) and generate grounded answer with citations."""
    # 1. Retrieve hybrid context
    retrieval = hybrid_search(
        query=req.query,
        mode=req.mode,
        top_k_chunks=req.top_k_chunks,
        max_graph_hops=req.max_graph_hops,
        document_id=req.document_id,
    )

    # 2. Generate grounded answer
    answer = generate_answer(
        query=req.query,
        retrieval=retrieval,
    )

    # 3. Format sources and visual subgraph
    sources = [
        SourceCitation(
            source_type=s.source_type,
            id=s.id,
            title=s.title,
            content=s.content,
            score=round(s.score, 4),
            filename=s.filename,
            chunk_index=s.chunk_index,
        )
        for s in retrieval.sources
    ]

    subgraph_nodes = [
        GraphNodeItem(**n) for n in retrieval.subgraph.get("nodes", [])
    ]
    subgraph_edges = [
        GraphEdgeItem(**e) for e in retrieval.subgraph.get("edges", [])
    ]

    return QueryResponse(
        query=req.query,
        mode=req.mode,
        answer=answer,
        sources=sources,
        graph_triples=retrieval.triples,
        subgraph=SubgraphData(
            nodes=subgraph_nodes,
            edges=subgraph_edges,
            triples=retrieval.triples,
        ),
        raw_context=retrieval.formatted_context,
    )


@router.post("/chat", response_model=QueryResponse)
async def chat_assistant(req: QueryRequest) -> QueryResponse:
    """Chat endpoint alias for query_assistant."""
    return await query_assistant(req)
