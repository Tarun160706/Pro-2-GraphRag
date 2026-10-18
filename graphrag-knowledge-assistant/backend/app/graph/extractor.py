"""
Entity and Relationship Extractor for Knowledge Graph construction.

Uses rule-based NLP, regex patterns, and linguistic relation extraction.
Works completely offline on CPU with zero paid API or heavyweight model requirements.
Can be extended with an LLM-based extractor in future iterations.
"""

import logging
import re
from typing import Optional

from app.graph.models import Entity, ExtractionResult, Relation

logger = logging.getLogger(__name__)

# Stopwords to filter out trivial entity candidates
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "and", "or", "but", "if", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "it", "its", "their", "they", "them",
    "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
}

# Known entity types for classification
_KNOWN_PERSONS = {"albert einstein", "alan turing", "geoffrey hinton", "yann lecun", "yoshua bengio"}
_KNOWN_LOCATIONS = {"germany", "united states", "usa", "uk", "france", "canada"}
_KNOWN_DATA_SYSTEMS = {"wikidata", "dbpedia", "google knowledge graph", "sparql", "cypher", "sqlite", "chromadb"}
_KNOWN_TECHNOLOGIES = {
    "graph neural networks", "gnn", "gnns", "knowledge graphs", "kg", "kgs",
    "graphrag", "rag", "large language models", "llm", "llms",
    "message passing neural networks", "mpnn", "mpnns",
    "graph convolutional networks", "gcn", "graph attention networks", "gat",
    "graphsage", "sentence transformers", "deep learning", "artificial intelligence",
    "neural networks", "transformer", "bert",
}


def _infer_entity_type(name: str) -> str:
    """Infer the entity category based on known heuristics."""
    clean = name.lower().strip()
    if clean in _KNOWN_PERSONS:
        return "PERSON"
    if clean in _KNOWN_LOCATIONS:
        return "LOCATION"
    if clean in _KNOWN_DATA_SYSTEMS:
        return "DATA_SYSTEM"
    if clean in _KNOWN_TECHNOLOGIES:
        return "TECHNOLOGY"
    if any(suffix in clean for suffix in ["network", "system", "algorithm", "model", "architecture", "framework"]):
        return "TECHNOLOGY"
    if any(suffix in clean for suffix in ["science", "learning", "reasoning", "retrieval", "theory"]):
        return "CONCEPT"
    return "CONCEPT"


def _clean_entity(text: str) -> str:
    """Clean extracted entity candidate string."""
    clean = re.sub(r"[^\w\s-]", "", text).strip()
    # Normalize multiple whitespace
    clean = re.sub(r"\s+", " ", clean)
    return clean


class RuleBasedExtractor:
    """Extracts entities and relationships from text using regex and linguistic rules."""

    def extract(
        self,
        text: str,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract entities and relations from a text passage."""
        entities_dict: dict[str, Entity] = {}
        relations_list: list[Relation] = []

        def add_entity(name: str, etype: Optional[str] = None, desc: str = "") -> Optional[str]:
            cleaned = _clean_entity(name)
            if not cleaned or len(cleaned) < 2 or cleaned.lower() in _STOPWORDS:
                return None
            key = cleaned.lower()
            if key not in entities_dict:
                category = etype or _infer_entity_type(cleaned)
                entities_dict[key] = Entity(
                    name=cleaned,
                    entity_type=category,
                    description=desc,
                    document_id=document_id,
                    chunk_id=chunk_id,
                )
            return entities_dict[key].name

        def add_relation(src: str, tgt: str, rel: str, weight: float = 1.0) -> None:
            clean_src = add_entity(src)
            clean_tgt = add_entity(tgt)
            if clean_src and clean_tgt and clean_src.lower() != clean_tgt.lower():
                relations_list.append(Relation(
                    source=clean_src,
                    target=clean_tgt,
                    relation=rel,
                    weight=weight,
                    document_id=document_id,
                    chunk_id=chunk_id,
                ))

        # 1. Match explicit triples: (Subject, Predicate, Object)
        triple_matches = re.findall(
            r"\(\s*([A-Za-z0-9\s\-]+?)\s*,\s*([A-Za-z0-9\s\-_]+?)\s*,\s*([A-Za-z0-9\s\-]+?)\s*\)",
            text,
        )
        for sub, pred, obj in triple_matches:
            add_relation(sub, obj, pred.upper().replace(" ", "_"), weight=1.0)

        # 2. Extract pattern: <X> is a/an <Y>
        is_a_matches = re.findall(
            r"([A-Z][A-Za-z0-9\s\-]{2,30}?)\s+(?:is|was)\s+(?:a|an)\s+([A-Za-z0-9\s\-]{3,40}?)(?=[,.;\n]|\s+that|\s+which)",
            text,
        )
        for sub, obj in is_a_matches:
            add_relation(sub, obj, "IS_A", weight=0.9)

        # 3. Extract pattern: <X> extends / extend <Y>
        extends_matches = re.findall(
            r"([A-Z][A-Za-z0-9\s\-]{2,30}?)\s+(?:extends|extend)\s+([A-Za-z0-9\s\-]{3,40}?)(?=[,.;\n]|\s+to|\s+by)",
            text,
        )
        for sub, obj in extends_matches:
            add_relation(sub, obj, "EXTENDS", weight=0.95)

        # 4. Extract pattern: <X> combines / combine <Y> with / and <Z>
        combines_matches = re.findall(
            r"([A-Z][A-Za-z0-9\s\-]{2,30}?)\s+(?:combines|combine)\s+([A-Za-z0-9\s\-]{3,35}?)\s+(?:with|and)\s+([A-Za-z0-9\s\-]{3,35}?)(?=[,.;\n])",
            text,
        )
        for sub, obj1, obj2 in combines_matches:
            add_relation(sub, obj1, "COMBINES_WITH", weight=0.9)
            add_relation(sub, obj2, "COMBINES_WITH", weight=0.9)

        # 5. Extract pattern: <X> includes / include / such as <Y>
        includes_matches = re.findall(
            r"([A-Z][A-Za-z0-9\s\-]{2,35}?)\s+(?:includes|include|such as)\s+([A-Za-z0-9\s\-,\(\)]+?)(?=[.;\n])",
            text,
        )
        for sub, obj_list_str in includes_matches:
            # Split items like "Wikidata, DBpedia, and Google Knowledge Graph"
            items = re.split(r",\s*|\s+and\s+|\s+or\s+", obj_list_str)
            for item in items:
                item_clean = item.strip()
                if item_clean and len(item_clean) > 2 and item_clean.lower() not in _STOPWORDS:
                    add_relation(sub, item_clean, "INCLUDES", weight=0.85)

        # 6. Extract pattern: <X> applied in / used for / used in <Y>
        applied_matches = re.findall(
            r"([A-Z][A-Za-z0-9\s\-]{2,30}?)\s+are\s+(?:applied in|used in|used for)\s+([A-Za-z0-9\s\-,\(\)]+?)(?=[.;\n])",
            text,
        )
        for sub, apps_str in applied_matches:
            items = re.split(r",\s*|\s+and\s+", apps_str)
            for item in items:
                item_clean = item.strip()
                if item_clean and len(item_clean) > 2:
                    add_relation(sub, item_clean, "APPLIED_IN", weight=0.85)

        # 7. Extract pattern: <X> excels at / specializes in <Y>
        excels_matches = re.findall(
            r"([A-Z][A-Za-z0-9\s\-]{2,30}?)\s+(?:excel at|excels at|specializes in)\s+([A-Za-z0-9\s\-,\(\)]+?)(?=[.;\n])",
            text,
        )
        for sub, feats_str in excels_matches:
            items = re.split(r",\s*|\s+and\s+", feats_str)
            for item in items:
                item_clean = item.strip()
                if item_clean and len(item_clean) > 2:
                    add_relation(sub, item_clean, "EXCELS_AT", weight=0.85)

        # 8. Extract Prominent Named Entities
        # 8a. Extract Candidate / Person Name (especially from resume headers)
        name_match = re.search(r"^\s*([A-Z][A-Z\s]{1,30}[A-Z])\s*(?:\+\d|[\w\.-]+@|\|)", text, re.MULTILINE)
        if not name_match:
            name_match = re.search(r"(?:Name|Candidate):\s*([A-Za-z\s]+)", text, re.IGNORECASE)
        if name_match:
            candidate_name = name_match.group(1).strip().title()
            add_entity(candidate_name, etype="PERSON", desc="Candidate / Document Subject")

        # 8b. Match multi-word capitalized phrases like "Knowledge Graph", "Machine Learning"
        cap_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
        for phrase in cap_phrases:
            add_entity(phrase)

        # 8c. Match technical acronyms (filter out non-tech abbreviations & personal names)
        _NON_TECH_ACRONYMS = {
            "CGPA", "GPA", "VHN", "SRM", "SRMMCET", "HSC", "SSLC", "DOB", "ID", "NO", "KNOWN", "REF", "PDF", "DOC",
            "TARUN", "FEBRUARY", "JANUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
        }
        acronyms = re.findall(r"\b[A-Z]{2,6}s?\b", text)
        for acr in acronyms:
            if acr.upper() not in _NON_TECH_ACRONYMS and acr.lower() not in _STOPWORDS:
                add_entity(acr, etype="TECHNOLOGY")

        # 9. Co-occurrence fallback: link prominent entities within the same sentence
        # if they don't already have direct edges
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            sentence_entities = []
            for key, entity in entities_dict.items():
                if entity.name.lower() in sentence.lower():
                    sentence_entities.append(entity.name)

            if len(sentence_entities) >= 2:
                # Link adjacent prominent entities
                for i in range(len(sentence_entities) - 1):
                    e1, e2 = sentence_entities[i], sentence_entities[i + 1]
                    # Only add if no relation already exists
                    existing = [r for r in relations_list if (r.source == e1 and r.target == e2) or (r.source == e2 and r.target == e1)]
                    if not existing:
                        add_relation(e1, e2, "RELATED_TO", weight=0.5)

        logger.debug(
            "Extracted %d entities and %d relations from chunk %s",
            len(entities_dict), len(relations_list), chunk_id,
        )

        return ExtractionResult(
            entities=list(entities_dict.values()),
            relations=relations_list,
        )


# Singleton extractor instance
_extractor = RuleBasedExtractor()


def extract_knowledge(
    text: str,
    document_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> ExtractionResult:
    """Public interface for entity and relation extraction."""
    return _extractor.extract(text, document_id, chunk_id)
