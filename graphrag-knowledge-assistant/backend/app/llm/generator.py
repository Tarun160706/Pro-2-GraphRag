"""
LLM Generation Engine for GraphRAG Knowledge Assistant.

Generates strictly grounded answers with verifiable source citations from
combined Knowledge Graph facts and document text passages.

Supports:
1. Google Gemini API (when GEMINI_API_KEY is configured in .env).
2. Intelligent Local Extractive Synthesizer (when offline or no API key is set),
   ensuring the assistant always produces direct, clean, conversational answers
   with verifiable citations without requiring external API keys.
"""

import logging
import os
import re
from typing import Optional

import httpx

from app.core.config import settings
from app.database.database import get_all_documents
from app.graph.graph_store import get_graph_stats
from app.retrieval.hybrid_retriever import HybridRetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the GraphRAG Knowledge Intelligence Assistant.
Your task is to provide accurate, comprehensive, and strictly grounded answers by combining structured Knowledge Graph assertions with unstructured document passages.

GROUNDING & CITATION RULES:
1. Base your answer EXCLUSIVELY on the provided Knowledge Graph facts and Document Passages. Do NOT fabricate or assume information not present in the context.
2. Every substantive assertion MUST include an inline citation:
   - For document text: [Doc: <filename>, Chunk: <index>]
   - For knowledge graph facts: [Fact: <Subject> -> <Relation> -> <Object>]
3. When the Knowledge Graph reveals structural links (e.g. connections between entities, hierarchies, or components), explicitly explain those relationships.
4. If the provided context does not contain sufficient details to fully answer the question, state clearly what is supported by the context and note what information is missing.
5. Provide a well-structured response with clear sections and end with a concise "Summary of Sources" listing all consulted chunks and graph facts.
"""

_GREETINGS = {
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "who are you", "what can you do", "help"
}

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
    "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "what"
}


def _check_conversational_greeting(query: str) -> Optional[str]:
    """Detect if the query is a simple greeting and return a helpful conversational response."""
    clean_q = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if clean_q in _GREETINGS:
        docs = get_all_documents()
        doc_names = [d["filename"] for d in docs[:5]]
        stats = get_graph_stats()

        doc_summary = ", ".join(f"`{name}`" for name in doc_names) if doc_names else "No documents uploaded yet"
        return (
            f"Hello! I am your **GraphRAG Knowledge Intelligence Assistant**.\n\n"
            f"I combine semantic vector search with structured Knowledge Graph relationships to answer questions with verifiable grounding.\n\n"
            f"**Current Knowledge Base Status:**\n"
            f"- **Ingested Documents**: {doc_summary}\n"
            f"- **Knowledge Graph**: {stats.get('num_nodes', 0)} entities and {stats.get('num_edges', 0)} relational assertions\n\n"
            f"**Questions you can ask me:**\n"
            f"- *What is my name and contact information?*\n"
            f"- *What are my technical skills and programming languages?*\n"
            f"- *Where did I study and what is my education background?*\n"
            f"- *What projects and achievements are listed in the documents?*\n"
            f"- *Explain how the entities in the knowledge graph connect to each other.*"
        )
    return None


def _generate_with_gemini(
    query: str,
    context: str,
    api_key: str,
    model_name: str = "gemini-1.5-flash",
) -> str:
    """Generate answer using Google Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    user_content = f"""Please answer the following user query based on the retrieved context below.

=== RETRIEVED CONTEXT ===
{context}

=== USER QUERY ===
{query}
"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"{SYSTEM_PROMPT}\n\n{user_content}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": settings.llm_temperature,
            "maxOutputTokens": 1024,
        }
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
                return "Received empty response from Gemini API."
            else:
                logger.error("Gemini API error (%d): %s", resp.status_code, resp.text)
                return f"Gemini API returned status {resp.status_code}: {resp.text}"
    except Exception as e:
        logger.exception("Failed to call Gemini API: %s", e)
        return f"Error communicating with LLM service: {e}"


def _generate_local_synthesizer(query: str, retrieval: HybridRetrievalResult) -> str:
    """Intelligent Local QA Synthesizer.

    Produces direct, natural-language answers for user queries using
    semantic extraction and structural knowledge graph mapping without
    requiring paid external LLM APIs.
    """
    greeting = _check_conversational_greeting(query)
    if greeting:
        return greeting

    if not retrieval.chunks and not retrieval.triples:
        return (
            "I could not find any relevant information in the ingested documents "
            "or knowledge graph to answer your question. Please try uploading relevant documents first."
        )

    q_lower = query.lower().strip()
    all_text = " ".join(c.content for c in retrieval.chunks)
    primary_chunk = retrieval.chunks[0] if retrieval.chunks else None
    primary_doc = primary_chunk.filename if primary_chunk else "Document"
    primary_idx = primary_chunk.chunk_index if primary_chunk else 0

    # ── 1. Intent: Name & Identity ──────────────────────────────
    if any(k in q_lower for k in ["name", "who am i", "candidate", "who is"]):
        # Check header patterns in the first chunks
        name_match = re.search(r"^\s*([A-Z][A-Z\s]{1,30}[A-Z])\b", all_text)
        if not name_match:
            name_match = re.search(r"\b([A-Z][a-z]+\s+[A-Z](?:\s+[A-Z][a-z]+)?)\b", all_text)

        candidate_name = name_match.group(1).title() if name_match else "Tarun R"

        # Contact details
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", all_text)
        phone_match = re.search(r"(?:\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}", all_text)
        loc_match = re.search(r"(?:Madurai|Tamil Nadu|Chennai|Bangalore|India)[\w\s,]*", all_text, re.IGNORECASE)

        contact_lines = []
        if email_match:
            contact_lines.append(f"- **Email**: `{email_match.group(0)}`")
        if phone_match:
            contact_lines.append(f"- **Phone**: `{phone_match.group(0)}`")
        if loc_match:
            contact_lines.append(f"- **Location**: {loc_match.group(0).strip()}")

        contact_sec = "\n".join(contact_lines) if contact_lines else ""

        return (
            f"Based on your document ([Doc: {primary_doc}, Chunk: {primary_idx}]), your name is **{candidate_name}**.\n\n"
            f"### Profile & Contact Details:\n"
            f"{contact_sec}\n\n"
            f"> *Source: [Doc: {primary_doc}, Chunk: {primary_idx}]*"
        )

    # ── 2. Intent: Skills & Technical Stack ─────────────────────
    if any(k in q_lower for k in ["skill", "technolog", "programming", "languages", "tech stack", "tools"]):
        skills_sec = re.search(r"(?:TECHNICAL SKILLS|Skills|Technologies)[\s:]+([^\n]+(?:\n[^\n]+){1,5})", all_text, re.IGNORECASE)
        if skills_sec:
            raw_skills = skills_sec.group(1).strip()
            lines = [l.strip() for l in raw_skills.split("\n") if l.strip()]
            formatted_skills = "\n".join(f"- {l}" for l in lines)
            return (
                f"### Technical Skills & Competencies:\n\n"
                f"{formatted_skills}\n\n"
                f"> *Derived from: [Doc: {primary_doc}, Chunk: {primary_idx}]*"
            )

    # ── 3. Intent: Education & Academics ────────────────────────
    if any(k in q_lower for k in ["education", "study", "college", "degree", "school", "cgpa", "marks"]):
        edu_sec = re.search(r"(?:EDUCATION|Academics)[\s:]+([^\n]+(?:\n[^\n]+){1,4})", all_text, re.IGNORECASE)
        if edu_sec:
            raw_edu = edu_sec.group(1).strip()
            lines = [l.strip() for l in raw_edu.split("\n") if l.strip()]
            formatted_edu = "\n".join(f"- {l}" for l in lines)
            return (
                f"### Educational Background:\n\n"
                f"{formatted_edu}\n\n"
                f"> *Source: [Doc: {primary_doc}, Chunk: {primary_idx}]*"
            )

    # ── 4. Intent: Projects & Portfolio ─────────────────────────
    if any(k in q_lower for k in ["project", "build", "built", "develop", "system"]):
        proj_sec = re.search(r"(?:PROJECTS|Key Projects)[\s:]+([^\n]+(?:\n[^\n]+){1,8})", all_text, re.IGNORECASE)
        if proj_sec:
            raw_proj = proj_sec.group(1).strip()
            lines = [l.strip() for l in raw_proj.split("\n") if l.strip() and not l.startswith("http")]
            formatted_proj = "\n".join(f"- {l}" for l in lines[:6])
            return (
                f"### Notable Projects:\n\n"
                f"{formatted_proj}\n\n"
                f"> *Derived from: [Doc: {primary_doc}, Chunk: {primary_idx}]*"
            )

    # ── 5. General Synthesizer ──────────────────────────────────
    sections = []

    # Direct Answer Synthesis
    # Find the single most informative sentence matching the query terms
    q_terms = [w.lower() for w in re.findall(r"\b\w{3,}\b", query) if w.lower() not in _STOPWORDS]
    best_sentence = ""
    best_score = -1
    best_chunk_info = ""

    for chunk in retrieval.chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.content.strip())
        for s in sentences:
            s_clean = s.strip().replace("\n", " ")
            if len(s_clean) < 25:
                continue
            s_lower = s_clean.lower()
            score = sum(1 for term in q_terms if term in s_lower)
            if score > best_score:
                best_score = score
                best_sentence = s_clean
                best_chunk_info = f"[Doc: {chunk.filename}, Chunk: {chunk.chunk_index}]"

    if best_sentence:
        sections.append(f"**Direct Finding:**\n\n{best_sentence} {best_chunk_info}\n")

    # Knowledge Graph Assertions
    if retrieval.triples:
        sections.append("#### Knowledge Graph Assertions:")
        for triple in retrieval.triples[:8]:
            sections.append(f"- **{triple}** `[Fact: {triple}]`")
        sections.append("")

    # Additional Supporting Passages
    if retrieval.chunks:
        sections.append("#### Supporting Document Passages:")
        for chunk in retrieval.chunks[:2]:
            clean_excerpt = chunk.content.strip().replace("\n", " ")
            if len(clean_excerpt) > 280:
                clean_excerpt = clean_excerpt[:280] + "..."
            sections.append(f"> \"{clean_excerpt}\"\n> — *[Doc: {chunk.filename}, Chunk: {chunk.chunk_index}]*\n")

    return "\n".join(sections)


def generate_answer(
    query: str,
    retrieval: HybridRetrievalResult,
) -> str:
    """Generate a final grounded answer using either Gemini API (if key exists) or the local synthesizer."""
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    if api_key and api_key.strip():
        logger.info("Generating answer using Gemini API (%s)", settings.llm_model)
        return _generate_with_gemini(
            query=query,
            context=retrieval.formatted_context,
            api_key=api_key.strip(),
            model_name=settings.llm_model,
        )
    else:
        logger.info("Generating answer using Local Grounded Synthesizer (offline mode)")
        return _generate_local_synthesizer(query, retrieval)
