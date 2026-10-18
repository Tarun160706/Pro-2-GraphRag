"""
Knowledge Graph API endpoints.

GET  /graph/stats               — Knowledge Graph size and metadata
GET  /graph/nodes               — List all extracted entities/nodes
GET  /graph/edges               — List all relationships/edges
POST /graph/query               — Query graph by keyword/entity, return k-hop facts & subgraph
GET  /graph/subgraph/{entity}   — Retrieve visual neighborhood for a specific entity
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.database.database import get_all_graph_edges, get_all_graph_nodes
from app.graph.graph_store import get_graph_stats, get_subgraph, search_nodes
from app.retrieval.graph_retriever import retrieve_graph_context
from app.schemas.schemas import (
    GraphEdgeItem,
    GraphNodeItem,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphStatsResponse,
    SubgraphData,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats() -> GraphStatsResponse:
    """Return summary statistics of the current Knowledge Graph."""
    stats = get_graph_stats()
    return GraphStatsResponse(**stats)


@router.get("/nodes", response_model=list[GraphNodeItem])
async def list_nodes(
    document_id: Optional[str] = Query(None, description="Filter nodes by document ID"),
    search: Optional[str] = Query(None, description="Search nodes by name substring"),
) -> list[GraphNodeItem]:
    """List knowledge graph entities, optionally filtered by document or search term."""
    if search:
        matches = search_nodes(search, limit=50)
        return [GraphNodeItem(**m) for m in matches]

    nodes = get_all_graph_nodes(document_id=document_id)
    return [
        GraphNodeItem(
            id=n.get("id"),
            name=n["name"],
            entity_type=n.get("entity_type", "CONCEPT"),
            description=n.get("description", ""),
        )
        for n in nodes
    ]


@router.get("/edges", response_model=list[GraphEdgeItem])
async def list_edges(
    document_id: Optional[str] = Query(None, description="Filter edges by document ID"),
) -> list[GraphEdgeItem]:
    """List knowledge graph relationships, optionally filtered by document."""
    edges = get_all_graph_edges(document_id=document_id)
    return [
        GraphEdgeItem(
            source=e["source"],
            target=e["target"],
            relation=e["relation"],
            weight=e.get("weight", 1.0),
        )
        for e in edges
    ]


@router.post("/query", response_model=GraphQueryResponse)
async def query_graph(req: GraphQueryRequest) -> GraphQueryResponse:
    """Retrieve subgraph facts and visual relationships matching a natural language query."""
    result = retrieve_graph_context(
        query=req.query,
        max_hops=req.max_hops,
        max_nodes=req.max_nodes,
    )

    nodes = [GraphNodeItem(**n) for n in result.subgraph.get("nodes", [])]
    edges = [GraphEdgeItem(**e) for e in result.subgraph.get("edges", [])]

    return GraphQueryResponse(
        query=result.query,
        matched_entities=result.matched_entities,
        total_triples=len(result.triples),
        triples=result.triples,
        subgraph=SubgraphData(nodes=nodes, edges=edges, triples=result.triples),
        formatted_context=result.formatted_context,
    )


@router.get("/subgraph/{entity_name}", response_model=SubgraphData)
async def get_entity_subgraph(
    entity_name: str,
    max_hops: int = Query(default=1, ge=1, le=3),
    max_nodes: int = Query(default=50, ge=1, le=100),
) -> SubgraphData:
    """Retrieve the visual k-hop subgraph centered around a specific entity."""
    subgraph = get_subgraph(
        center_entities=[entity_name],
        max_hops=max_hops,
        max_nodes=max_nodes,
    )
    if not subgraph["nodes"]:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_name}' not found in knowledge graph.",
        )

    nodes = [GraphNodeItem(**n) for n in subgraph["nodes"]]
    edges = [GraphEdgeItem(**e) for e in subgraph["edges"]]
    return SubgraphData(nodes=nodes, edges=edges, triples=subgraph.get("triples", []))
