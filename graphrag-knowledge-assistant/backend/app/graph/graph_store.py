"""
Knowledge Graph store using NetworkX backed by SQLite and local JSON persistence.

Maintains an in-memory MultiDiGraph for fast multi-hop traversal, neighborhood queries,
and subgraph extraction, with automatic persistence to disk.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx

from app.core.config import settings
from app.database.database import (
    delete_graph_by_document,
    get_all_graph_edges,
    get_all_graph_nodes,
    insert_graph_edges,
    insert_graph_nodes,
)
from app.graph.models import Entity, Relation

logger = logging.getLogger(__name__)

# Global singleton in-memory graph
_graph: Optional[nx.MultiDiGraph] = None


def _get_graph_filepath() -> Path:
    """Return path to persistent graph JSON file."""
    graph_dir = Path(settings.graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    return graph_dir / "knowledge_graph.json"


def get_graph() -> nx.MultiDiGraph:
    """Get or lazily initialize the Knowledge Graph."""
    global _graph
    if _graph is None:
        _graph = nx.MultiDiGraph()
        filepath = _get_graph_filepath()

        # 1. Try loading from persistent JSON file
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _graph = nx.node_link_graph(data, directed=True, multigraph=True)
                logger.info(
                    "Loaded Knowledge Graph from %s (%d nodes, %d edges)",
                    filepath, _graph.number_of_nodes(), _graph.number_of_edges(),
                )
                return _graph
            except Exception as e:
                logger.warning("Could not load graph from %s: %s. Rebuilding from SQLite.", filepath, e)

        # 2. Fallback: Rebuild from SQLite records
        nodes = get_all_graph_nodes()
        edges = get_all_graph_edges()
        for n in nodes:
            _graph.add_node(
                n["name"],
                entity_type=n.get("entity_type", "CONCEPT"),
                description=n.get("description", ""),
                document_id=n.get("document_id", ""),
            )
        for e in edges:
            _graph.add_edge(
                e["source"],
                e["target"],
                relation=e.get("relation", "RELATED_TO"),
                weight=e.get("weight", 1.0),
                document_id=e.get("document_id", ""),
                chunk_id=e.get("chunk_id", ""),
            )
        logger.info(
            "Rebuilt Knowledge Graph from SQLite (%d nodes, %d edges)",
            _graph.number_of_nodes(), _graph.number_of_edges(),
        )
    return _graph


def save_graph() -> None:
    """Persist the in-memory graph to JSON."""
    g = get_graph()
    filepath = _get_graph_filepath()
    try:
        data = nx.node_link_data(g)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved Knowledge Graph to %s", filepath)
    except Exception as e:
        logger.error("Failed to save graph to %s: %s", filepath, e)


def add_entities_and_relations(
    entities: list[Entity],
    relations: list[Relation],
    document_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> tuple[int, int]:
    """Add entities and relations to the in-memory graph and persist to SQLite and JSON.

    Returns:
        (num_nodes_added, num_edges_added)
    """
    g = get_graph()

    # Prepare SQLite records
    nodes_db = []
    for ent in entities:
        doc_id = ent.document_id or document_id or "unknown"
        chk_id = ent.chunk_id or chunk_id
        g.add_node(
            ent.name,
            entity_type=ent.entity_type,
            description=ent.description,
            document_id=doc_id,
            chunk_id=chk_id,
        )
        nodes_db.append({
            "id": ent.name.lower(),
            "name": ent.name,
            "entity_type": ent.entity_type,
            "description": ent.description,
            "document_id": doc_id,
            "chunk_id": chk_id,
        })

    edges_db = []
    for rel in relations:
        doc_id = rel.document_id or document_id or "unknown"
        chk_id = rel.chunk_id or chunk_id
        g.add_edge(
            rel.source,
            rel.target,
            relation=rel.relation,
            weight=rel.weight,
            document_id=doc_id,
            chunk_id=chk_id,
        )
        edges_db.append({
            "id": f"{rel.source.lower()}-{rel.relation.upper()}-{rel.target.lower()}",
            "source": rel.source,
            "target": rel.target,
            "relation": rel.relation,
            "weight": rel.weight,
            "document_id": doc_id,
            "chunk_id": chk_id,
        })

    # Save to SQLite
    insert_graph_nodes(nodes_db)
    insert_graph_edges(edges_db)

    # Persist graph to file
    save_graph()

    logger.info(
        "Graph updated: %d nodes, %d edges total (+%d entities, +%d relations)",
        g.number_of_nodes(), g.number_of_edges(), len(entities), len(relations),
    )
    return len(entities), len(relations)


def get_graph_stats() -> dict:
    """Calculate and return Knowledge Graph statistics."""
    g = get_graph()
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()

    # Node types distribution
    node_types: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        t = data.get("entity_type", "CONCEPT")
        node_types[t] = node_types.get(t, 0) + 1

    # Edge relations distribution
    relation_types: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        r = data.get("relation", "RELATED_TO")
        relation_types[r] = relation_types.get(r, 0) + 1

    density = nx.density(g) if num_nodes > 1 else 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 4),
        "node_types": node_types,
        "relation_types": relation_types,
        "graph_file": str(_get_graph_filepath()),
    }


def search_nodes(query: str, limit: int = 10) -> list[dict]:
    """Find nodes matching query string (case-insensitive substring)."""
    g = get_graph()
    q = query.lower().strip()
    matches = []
    for node, data in g.nodes(data=True):
        if q in node.lower():
            degree = g.degree(node)
            matches.append({
                "name": node,
                "entity_type": data.get("entity_type", "CONCEPT"),
                "degree": degree,
                "description": data.get("description", ""),
            })
    # Sort by degree (centrality) descending
    matches.sort(key=lambda x: x["degree"], reverse=True)
    return matches[:limit]


def get_subgraph(
    center_entities: list[str],
    max_hops: int = 1,
    max_nodes: int = 50,
) -> dict:
    """Extract a k-hop subgraph around center entities.

    Returns:
        Dict formatted for D3 / Vis.js frontend visualization:
        {"nodes": [...], "edges": [...], "triples": [...]}
    """
    g = get_graph()
    if g.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "triples": []}

    # Normalize entity matching (case-insensitive)
    matched_nodes = set()
    node_name_map = {n.lower(): n for n in g.nodes()}

    for ent in center_entities:
        ent_clean = ent.lower().strip()
        if ent_clean in node_name_map:
            matched_nodes.add(node_name_map[ent_clean])
        else:
            # Substring match
            for low_name, orig_name in node_name_map.items():
                if ent_clean in low_name or low_name in ent_clean:
                    matched_nodes.add(orig_name)

    if not matched_nodes:
        return {"nodes": [], "edges": [], "triples": []}

    # Breadth-first search for k-hop neighborhood
    subgraph_nodes = set(matched_nodes)
    current_frontier = set(matched_nodes)

    for _ in range(max_hops):
        next_frontier = set()
        for node in current_frontier:
            if len(subgraph_nodes) >= max_nodes:
                break
            # Add successors and predecessors (directed)
            neighbors = set(g.successors(node)).union(set(g.predecessors(node)))
            for nbr in neighbors:
                if nbr not in subgraph_nodes and len(subgraph_nodes) < max_nodes:
                    next_frontier.add(nbr)
                    subgraph_nodes.add(nbr)
        current_frontier = next_frontier
        if len(subgraph_nodes) >= max_nodes:
            break

    # Build node and edge lists
    nodes_res = []
    for n in subgraph_nodes:
        data = g.nodes[n]
        nodes_res.append({
            "id": n,
            "name": n,
            "entity_type": data.get("entity_type", "CONCEPT"),
            "degree": g.degree(n),
            "is_query_match": n in matched_nodes,
        })

    edges_res = []
    triples = []
    for u, v, k, data in g.edges(subgraph_nodes, keys=True, data=True):
        if v in subgraph_nodes:
            rel = data.get("relation", "RELATED_TO")
            edges_res.append({
                "source": u,
                "target": v,
                "relation": rel,
                "weight": data.get("weight", 1.0),
            })
            triples.append(f"{u} -[{rel}]-> {v}")

    return {
        "nodes": nodes_res,
        "edges": edges_res,
        "triples": list(set(triples)),
    }


def rebuild_from_db() -> nx.MultiDiGraph:
    """Reconstruct in-memory graph directly from SQLite tables."""
    global _graph
    _graph = nx.MultiDiGraph()
    nodes = get_all_graph_nodes()
    edges = get_all_graph_edges()
    for n in nodes:
        _graph.add_node(
            n["name"],
            entity_type=n.get("entity_type", "CONCEPT"),
            description=n.get("description", ""),
            document_id=n.get("document_id", ""),
        )
    for e in edges:
        _graph.add_edge(
            e["source"],
            e["target"],
            relation=e.get("relation", "RELATED_TO"),
            weight=e.get("weight", 1.0),
            document_id=e.get("document_id", ""),
            chunk_id=e.get("chunk_id", ""),
        )
    save_graph()
    return _graph


def delete_by_document(document_id: str) -> None:
    """Remove nodes and edges originating from a document."""
    # 1. Remove from SQLite
    delete_graph_by_document(document_id)

    # 2. Reconstruct graph directly from SQLite and update JSON
    rebuild_from_db()
    logger.info("Purged graph elements for document %s", document_id)
