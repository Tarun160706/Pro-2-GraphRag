"""
Graph data structures for Knowledge Graph nodes, edges, and extraction results.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """A node in the Knowledge Graph."""
    name: str
    entity_type: str = "CONCEPT"
    description: str = ""
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.name.lower().strip(),
            "name": self.name.strip(),
            "entity_type": self.entity_type,
            "description": self.description,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Relation:
    """A directed edge in the Knowledge Graph connecting two entities."""
    source: str
    target: str
    relation: str
    weight: float = 1.0
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_dict(self) -> dict:
        src = self.source.lower().strip()
        tgt = self.target.lower().strip()
        rel = self.relation.upper().strip().replace(" ", "_")
        return {
            "id": f"{src}-{rel}-{tgt}",
            "source": self.source.strip(),
            "target": self.target.strip(),
            "relation": rel,
            "weight": self.weight,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
        }


@dataclass
class ExtractionResult:
    """Collection of extracted entities and relations from a text unit."""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
