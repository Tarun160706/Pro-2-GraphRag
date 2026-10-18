"""
Graph-based retrieval using Knowledge Graph traversal.

Given a user query, identifies relevant entities, traverses their k-hop
neighborhood in NetworkX, and formats structured facts/triples to augment
retrieval context.
"""

import logging
import re
from dataclasses import dataclass, field

from app.graph.extractor import extract_knowledge
from app.graph.graph_store import get_graph, get_subgraph

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    """Result of graph retrieval containing structured facts and visual subgraph."""
    query: str
    matched_entities: list[str]
    triples: list[str]
    subgraph: dict
    formatted_context: str = ""

    def __post_init__(self):
        if not self.formatted_context and self.triples:
            lines = ["### Knowledge Graph Facts:"]
            for triple in self.triples:
                lines.append(f"- {triple}")
            self.formatted_context = "\n".join(lines)


def retrieve_graph_context(
    query: str,
    max_hops: int = 1,
    max_nodes: int = 30,
) -> GraphRetrievalResult:
    """Retrieve graph facts and subgraph for a query.

    Args:
        query: User's question or search query.
        max_hops: Neighborhood traversal depth (1 or 2 hops).
        max_nodes: Maximum nodes in returned subgraph.

    Returns:
        GraphRetrievalResult with matched entities, triples, and formatted context.
    """
    g = get_graph()
    if g.number_of_nodes() == 0:
        return GraphRetrievalResult(
            query=query,
            matched_entities=[],
            triples=[],
            subgraph={"nodes": [], "edges": []},
            formatted_context="",
        )

    # 1. Identify entity candidates from query
    # A: Use rule-based extractor on query
    query_extracted = extract_knowledge(query)
    candidate_names = [e.name for e in query_extracted.entities]

    # B: Also search for existing graph node names that appear as substrings in the query
    q_lower = query.lower()
    for node in g.nodes():
        if len(node) > 2 and re.search(r"\b" + re.escape(node.lower()) + r"\b", q_lower):
            if node not in candidate_names:
                candidate_names.append(node)

    # C: If still no match, fallback to token matches (excluding common words)
    if not candidate_names:
        words = re.findall(r"\b[A-Za-z0-9\-]{3,}\b", query)
        for w in words:
            for node in g.nodes():
                if w.lower() in node.lower():
                    if node not in candidate_names:
                        candidate_names.append(node)

    # 2. Extract subgraph around matched entities
    subgraph_data = get_subgraph(
        center_entities=candidate_names,
        max_hops=max_hops,
        max_nodes=max_nodes,
    )

    matched_nodes = [
        n["name"] for n in subgraph_data.get("nodes", []) if n.get("is_query_match")
    ]
    triples = subgraph_data.get("triples", [])

    logger.info(
        "Graph retrieval for '%s': %d matched entities, %d triples",
        query[:50], len(matched_nodes), len(triples),
    )

    return GraphRetrievalResult(
        query=query,
        matched_entities=matched_nodes,
        triples=triples,
        subgraph={
            "nodes": subgraph_data.get("nodes", []),
            "edges": subgraph_data.get("edges", []),
        },
    )
