"""
Document management API endpoints.

POST /documents/upload   — Upload and ingest a document
GET  /documents          — List all documents
GET  /documents/{id}     — Get a single document
GET  /documents/{id}/chunks — Get all chunks for a document
DELETE /documents/{id}   — Delete a document and its chunks
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.database.database import (
    get_document,
    get_all_documents,
    get_chunks_by_document,
    delete_document,
)
from app.ingestion.pipeline import (
    process_document,
    SUPPORTED_EXTENSIONS,
    get_file_type,
)
from app.schemas.schemas import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    ChunkResponse,
    DocumentChunksResponse,
)
from app.vector.chroma_store import delete_by_document as delete_vectors_by_document
from app.graph.graph_store import delete_by_document as delete_graph_by_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a document file for ingestion.

    Supported formats: .pdf, .docx, .txt, .md
    """
    filename = file.filename or "unknown"
    file_type = get_file_type(filename)

    if file_type not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_type}'. "
                   f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Save uploaded file to the documents directory
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    save_path = docs_dir / filename

    # Avoid overwriting: append a suffix if file exists
    counter = 1
    while save_path.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        save_path = docs_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    file_size = save_path.stat().st_size
    logger.info("Saved uploaded file to %s (%d bytes)", save_path, file_size)

    # Run the ingestion pipeline
    try:
        result = process_document(
            file_path=str(save_path),
            filename=filename,
            file_size=file_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {e}",
        )

    return DocumentUploadResponse(**result)


@router.get("", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents."""
    docs = get_all_documents()
    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentResponse(**d) for d in docs],
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_by_id(document_id: str) -> DocumentResponse:
    """Get details for a specific document."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**doc)


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(document_id: str) -> DocumentChunksResponse:
    """Get all text chunks for a document."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = get_chunks_by_document(document_id)
    return DocumentChunksResponse(
        document_id=document_id,
        filename=doc["filename"],
        total_chunks=len(chunks),
        chunks=[ChunkResponse(**c) for c in chunks],
    )


@router.delete("/{document_id}", status_code=200)
async def delete_document_by_id(document_id: str) -> dict:
    """Delete a document, its chunks, vectors, and graph knowledge."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete vectors from ChromaDB
    delete_vectors_by_document(document_id)

    # Delete entities and relationships from Knowledge Graph
    delete_graph_by_document(document_id)

    # Delete the file from disk if it exists
    file_path = Path(doc["file_path"])
    if file_path.exists():
        file_path.unlink()
        logger.info("Deleted file %s", file_path)

    delete_document(document_id)
    return {"message": f"Document '{doc['filename']}' deleted successfully"}

