"""
Pydantic schemas for API request/response validation.

Separates the API contract from internal database representation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Document schemas ─────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Returned when listing or fetching a document."""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    num_chunks: int
    created_at: str
    updated_at: str


class DocumentUploadResponse(BaseModel):
    """Returned immediately after a file is uploaded."""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    message: str


class DocumentListResponse(BaseModel):
    """Wrapper for a list of documents."""
    total: int
    documents: list[DocumentResponse]


# ── Chunk schemas ────────────────────────────────────────────────

class ChunkResponse(BaseModel):
    """A single text chunk with its metadata."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class DocumentChunksResponse(BaseModel):
    """All chunks belonging to a document."""
    document_id: str
    filename: str
    total_chunks: int
    chunks: list[ChunkResponse]


# ── Search schemas ───────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request body for vector search."""
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    document_id: Optional[str] = Field(
        default=None, description="Restrict search to a specific document"
    )


class SearchResultItem(BaseModel):
    """A single search result with similarity score."""
    chunk_id: str
    content: str
    score: float
    document_id: str
    filename: str
    chunk_index: int


class SearchResponse(BaseModel):
    """Response wrapper for search results."""
    query: str
    total_results: int
    results: list[SearchResultItem]


# ── Graph schemas ────────────────────────────────────────────────

class GraphNodeItem(BaseModel):
    """Graph node item."""
    id: Optional[str] = None
    name: str
    entity_type: str = "CONCEPT"
    degree: Optional[int] = 0
    description: Optional[str] = ""
    is_query_match: Optional[bool] = False


class GraphEdgeItem(BaseModel):
    """Graph edge item."""
    source: str
    target: str
    relation: str
    weight: Optional[float] = 1.0


class SubgraphData(BaseModel):
    """Visual subgraph data containing nodes and edges."""
    nodes: list[GraphNodeItem]
    edges: list[GraphEdgeItem]
    triples: Optional[list[str]] = []


class GraphStatsResponse(BaseModel):
    """Overall statistics for the Knowledge Graph."""
    num_nodes: int
    num_edges: int
    density: float
    node_types: dict[str, int]
    relation_types: dict[str, int]
    graph_file: str


class GraphQueryRequest(BaseModel):
    """Request body for graph traversal query."""
    query: str = Field(..., min_length=1, description="Question or entity name to search")
    max_hops: int = Field(default=1, ge=1, le=3, description="Neighborhood traversal depth")
    max_nodes: int = Field(default=30, ge=1, le=100, description="Max nodes to include")


class GraphQueryResponse(BaseModel):
    """Response for graph query containing extracted triples and visual subgraph."""
    query: str
    matched_entities: list[str]
    total_triples: int
    triples: list[str]
    subgraph: SubgraphData
    formatted_context: str


# ── Full RAG Query / Chat schemas ────────────────────────────────

class QueryRequest(BaseModel):
    """Request for answer generation using GraphRAG."""
    query: str = Field(..., min_length=1, description="Question for the knowledge assistant")
    mode: str = Field(
        default="hybrid",
        description="Retrieval mode: 'hybrid', 'vector_only', or 'graph_only'",
    )
    top_k_chunks: int = Field(default=3, ge=1, le=10, description="Max text chunks to retrieve")
    max_graph_hops: int = Field(default=2, ge=1, le=3, description="Graph traversal depth")
    document_id: Optional[str] = Field(default=None, description="Optional document ID filter")


class SourceCitation(BaseModel):
    """A citation source supporting both chunks and graph assertions."""
    source_type: str
    id: str
    title: str
    content: str
    score: float
    filename: Optional[str] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    """Full grounded answer response from GraphRAG."""
    query: str
    mode: str
    answer: str
    sources: list[SourceCitation]
    graph_triples: list[str]
    subgraph: SubgraphData
    raw_context: str



