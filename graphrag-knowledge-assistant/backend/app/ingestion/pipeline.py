"""
Document ingestion pipeline.

Orchestrates: file saving → text extraction → chunking →
database storage → embedding → vector store.
This module is called by the API router and is the single entry point
for processing uploaded documents.
"""

import json
import logging
import uuid
from pathlib import Path

from app.core.config import settings
from app.database.database import (
    insert_document,
    update_document_status,
    insert_chunks,
)
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.ingestion.docx_loader import extract_text_from_docx
from app.ingestion.text_loader import extract_text_from_txt
from app.ingestion.chunker import chunk_text
from app.embeddings.embedding_service import embed_texts
from app.vector.chroma_store import add_embeddings
from app.graph.extractor import extract_knowledge
from app.graph.graph_store import add_entities_and_relations

logger = logging.getLogger(__name__)

# Map file extensions to their extraction functions
_EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".md": extract_text_from_txt,
}

SUPPORTED_EXTENSIONS = set(_EXTRACTORS.keys())


def get_file_type(filename: str) -> str:
    """Return the lowercased file extension (e.g. '.pdf')."""
    return Path(filename).suffix.lower()


def process_document(
    file_path: str,
    filename: str,
    file_size: int,
) -> dict:
    """Run the full ingestion pipeline for a single document.

    Steps:
      1. Register document in the database as 'processing'.
      2. Extract text using the appropriate loader.
      3. Chunk the text.
      4. Store chunks in SQLite.
      5. Embed chunks and store vectors in ChromaDB.
      6. Update document status to 'ingested'.

    Returns a dict with document id, filename, chunk count, and status.
    Raises on any failure and marks the document as 'failed'.
    """
    file_type = get_file_type(filename)
    doc_id = str(uuid.uuid4())

    # Step 1: Register
    insert_document(
        doc_id=doc_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        file_path=file_path,
    )
    logger.info("Registered document %s (%s)", doc_id, filename)

    try:
        # Step 2: Extract
        extractor = _EXTRACTORS.get(file_type)
        if extractor is None:
            raise ValueError(f"Unsupported file type: {file_type}")

        text = extractor(file_path)

        # Step 3: Chunk
        text_chunks = chunk_text(text)

        # Step 4: Store chunks in SQLite
        chunk_ids = [str(uuid.uuid4()) for _ in text_chunks]
        chunk_records = [
            {
                "id": chunk_ids[i],
                "document_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": json.dumps({
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                }),
            }
            for i, chunk in enumerate(text_chunks)
        ]
        insert_chunks(chunk_records)

        # Step 5: Embed and store in ChromaDB
        chunk_contents = [chunk.content for chunk in text_chunks]
        embeddings = embed_texts(chunk_contents)

        chroma_metadatas = [
            {
                "document_id": doc_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in text_chunks
        ]
        add_embeddings(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_contents,
            metadatas=chroma_metadatas,
        )

        # Step 6: Extract Knowledge Graph entities & relations
        all_entities = []
        all_relations = []
        for i, chunk in enumerate(text_chunks):
            ext_res = extract_knowledge(
                text=chunk.content,
                document_id=doc_id,
                chunk_id=chunk_ids[i],
            )
            all_entities.extend(ext_res.entities)
            all_relations.extend(ext_res.relations)

        num_entities, num_relations = add_entities_and_relations(
            entities=all_entities,
            relations=all_relations,
            document_id=doc_id,
        )

        # Step 7: Mark as ingested
        num_chunks = len(chunk_records)
        update_document_status(doc_id, "ingested", num_chunks=num_chunks)
        logger.info(
            "Document %s ingested: %d chunks, %d entities, %d relations from %s",
            doc_id, num_chunks, num_entities, num_relations, filename,
        )

        return {
            "id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "status": "ingested",
            "num_chunks": num_chunks,
            "message": f"Successfully ingested {filename} into {num_chunks} chunks.",
        }

    except Exception:
        update_document_status(doc_id, "failed")
        logger.exception("Failed to ingest %s", filename)
        raise

