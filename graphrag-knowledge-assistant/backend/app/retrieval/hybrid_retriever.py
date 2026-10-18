"""
Hybrid Retriever combining Vector Search and Knowledge Graph traversal.

Executes both dense semantic vector retrieval and structured knowledge graph
neighborhood traversal, then merges and re-ranks results using Reciprocal
Rank Fusion (RRF) and graph provenance matching.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.database.database import get_connection
from app.retrieval.graph_retriever import GraphRetrievalResult, retrieve_graph_context
from app.retrieval.vector_retriever import VectorSearchResult, vector_search

logger = logging.getLogger(__name__)


@dataclass
class HybridSource:
    """A citation source supporting both text chunks and graph facts."""
    source_type: str  # "chunk" | "graph_triple"
    id: str
    title: str
    content: str
    score: float
    filename: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class HybridRetrievalResult:
    """Unified retrieval result containing ranked chunks, graph triples, visual subgraph, and prompt context."""
    query: str
    mode: str  # "hybrid" | "vector_only" | "graph_only"
    chunks: list[VectorSearchResult] = field(default_factory=list)
    triples: list[str] = field(default_factory=list)
    subgraph: dict = field(default_factory=dict)
    sources: list[HybridSource] = field(default_factory=list)
    formatted_context: str = ""


def _get_chunk_ids_for_graph_entities(entity_names: list[str]) -> dict[str, int]:
    """Find chunk IDs in SQLite that contain the matched graph entities and count their mentions."""
    if not entity_names:
        return {}

    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in entity_names)
        query = f"""
            SELECT chunk_id, COUNT(*) as cnt
            FROM graph_nodes
            WHERE lower(name) IN ({placeholders}) AND chunk_id IS NOT NULL
            GROUP BY chunk_id
        """
        rows = conn.execute(query, [name.lower() for name in entity_names]).fetchall()
        return {r["chunk_id"]: r["cnt"] for r in rows}
    except Exception as e:
        logger.warning("Error fetching chunk provenance for entities: %s", e)
        return {}
    finally:
        conn.close()


def hybrid_search(
    query: str,
    mode: str = "hybrid",
    top_k_chunks: int = 3,
    max_graph_hops: int = 2,
    max_graph_nodes: int = 25,
    document_id: Optional[str] = None,
) -> HybridRetrievalResult:
    """Perform hybrid retrieval combining vector similarity and knowledge graph facts.

    Args:
        query: User natural language search query.
        mode: "hybrid" (default), "vector_only", or "graph_only".
        top_k_chunks: Number of vector chunks to return.
        max_graph_hops: Graph neighborhood traversal depth.
        max_graph_nodes: Max nodes to include in visual subgraph.
        document_id: Optional document filter.

    Returns:
        HybridRetrievalResult with combined context, sources, and visual subgraph.
    """
    vector_results: list[VectorSearchResult] = []
    graph_result = GraphRetrievalResult(
        query=query,
        matched_entities=[],
        triples=[],
        subgraph={"nodes": [], "edges": []},
        formatted_context="",
    )

    # 1. Vector Retrieval (if hybrid or vector_only)
    if mode in ("hybrid", "vector_only"):
        vector_results = vector_search(
            query=query,
            top_k=top_k_chunks,
            document_id=document_id,
        )

    # 2. Graph Retrieval (if hybrid or graph_only)
    if mode in ("hybrid", "graph_only"):
        graph_result = retrieve_graph_context(
            query=query,
            max_hops=max_graph_hops,
            max_nodes=max_graph_nodes,
        )

    # 3. Reciprocal Rank Fusion & Re-ranking for hybrid mode
    final_chunks = list(vector_results)
    if mode == "hybrid" and vector_results and graph_result.matched_entities:
        # Cross-reference: find chunks that contained the matched graph entities
        graph_chunk_counts = _get_chunk_ids_for_graph_entities(graph_result.matched_entities)

        # Apply Reciprocal Rank Fusion (RRF)
        # RRF score = w_vec / (k + rank_vec) + w_graph / (k + rank_graph)
        rrf_k = 60.0
        w_vec = 0.65
        w_graph = 0.35

        scored_chunks = []
        for rank, chunk in enumerate(vector_results):
            vec_rrf = w_vec / (rrf_k + rank + 1)
            # If this chunk contributed to the graph entities, give it a graph boost
            graph_mentions = graph_chunk_counts.get(chunk.chunk_id, 0)
            graph_rrf = (w_graph / rrf_k) * min(graph_mentions, 3)
            fused_score = vec_rrf + graph_rrf
            scored_chunks.append((fused_score, chunk))

        # Re-sort chunks by fused score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        final_chunks = [item[1] for item in scored_chunks]

    # 4. Compile Structured Sources for Citation
    sources: list[HybridSource] = []

    # Add text chunk sources
    for chunk in final_chunks:
        sources.append(HybridSource(
            source_type="chunk",
            id=chunk.chunk_id,
            title=f"{chunk.filename} (Chunk #{chunk.chunk_index})",
            content=chunk.content,
            score=chunk.score,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
        ))

    # Add graph triple sources
    for i, triple in enumerate(graph_result.triples):
        sources.append(HybridSource(
            source_type="graph_triple",
            id=f"triple-{i}",
            title="Knowledge Graph Assertion",
            content=triple,
            score=1.0,
        ))

    # 5. Build Unified Formatted Context for LLM Grounding
    context_sections = []

    if graph_result.triples:
        context_sections.append("### KNOWLEDGE GRAPH ASSERTIONS (VERIFIED FACTS):")
        for triple in graph_result.triples:
            context_sections.append(f"- [Fact] {triple}")

    if final_chunks:
        context_sections.append("\n### DOCUMENT PASSAGES:")
        for chunk in final_chunks:
            context_sections.append(
                f"--- [Document: {chunk.filename}, Chunk: {chunk.chunk_index}] ---\n{chunk.content}\n"
            )

    formatted_context = "\n".join(context_sections)

    logger.info(
        "Hybrid retrieval finished: mode=%s, %d chunks, %d graph triples, %d total sources",
        mode, len(final_chunks), len(graph_result.triples), len(sources),
    )

    return HybridRetrievalResult(
        query=query,
        mode=mode,
        chunks=final_chunks,
        triples=graph_result.triples,
        subgraph=graph_result.subgraph,
        sources=sources,
        formatted_context=formatted_context,
    )
