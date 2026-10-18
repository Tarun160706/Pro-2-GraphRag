"""
SQLite database setup using Python's built-in sqlite3.

Keeps things simple — no ORM needed for metadata storage.
The database stores document and chunk metadata; actual text
content lives in the chunks table for retrieval.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    file_path   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    num_chunks  INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    start_char  INTEGER,
    end_char    INTEGER,
    metadata    TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'CONCEPT',
    description TEXT,
    document_id TEXT NOT NULL,
    chunk_id    TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    document_id TEXT NOT NULL,
    chunk_id    TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc ON graph_nodes(document_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_doc ON graph_edges(document_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
"""


def get_connection() -> sqlite3.Connection:
    """Open a connection to the SQLite database.

    Enables WAL mode for better concurrent read performance and
    turns on foreign key enforcement.
    """
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # access columns by name
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    conn = get_connection()
    try:
        conn.executescript(_CREATE_TABLES_SQL)
        conn.commit()
        logger.info("Database initialized at %s", settings.sqlite_db_path)
    finally:
        conn.close()


# ── Document CRUD ────────────────────────────────────────────────

def insert_document(
    doc_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    file_path: str,
) -> None:
    """Insert a new document record."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO documents (id, filename, file_type, file_size, file_path, status)
               VALUES (?, ?, ?, ?, ?, 'processing')""",
            (doc_id, filename, file_type, file_size, file_path),
        )
        conn.commit()
    finally:
        conn.close()


def update_document_status(
    doc_id: str,
    status: str,
    num_chunks: Optional[int] = None,
) -> None:
    """Update a document's processing status and optional chunk count."""
    conn = get_connection()
    try:
        if num_chunks is not None:
            conn.execute(
                """UPDATE documents
                   SET status = ?, num_chunks = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                (status, num_chunks, doc_id),
            )
        else:
            conn.execute(
                """UPDATE documents
                   SET status = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                (status, doc_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_document(doc_id: str) -> Optional[dict]:
    """Fetch a single document by ID. Returns None if not found."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_documents() -> list[dict]:
    """Fetch all documents, newest first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(doc_id: str) -> bool:
    """Delete a document and its chunks (CASCADE). Returns True if found."""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ── Chunk CRUD ───────────────────────────────────────────────────

def insert_chunks(chunks: list[dict]) -> None:
    """Bulk-insert chunk records.

    Each dict must have: id, document_id, chunk_index, content,
    start_char, end_char, metadata.
    """
    if not chunks:
        return
    conn = get_connection()
    try:
        conn.executemany(
            """INSERT INTO chunks (id, document_id, chunk_index, content,
                                   start_char, end_char, metadata)
               VALUES (:id, :document_id, :chunk_index, :content,
                        :start_char, :end_char, :metadata)""",
            chunks,
        )
        conn.commit()
    finally:
        conn.close()


def get_chunks_by_document(doc_id: str) -> list[dict]:
    """Fetch all chunks for a document, ordered by index."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Graph CRUD ────────────────────────────────────────────────────

def insert_graph_nodes(nodes: list[dict]) -> None:
    """Bulk-insert or ignore graph nodes."""
    if not nodes:
        return
    conn = get_connection()
    try:
        conn.executemany(
            """INSERT OR IGNORE INTO graph_nodes (id, name, entity_type, description, document_id, chunk_id)
               VALUES (:id, :name, :entity_type, :description, :document_id, :chunk_id)""",
            nodes,
        )
        conn.commit()
    finally:
        conn.close()


def insert_graph_edges(edges: list[dict]) -> None:
    """Bulk-insert graph edges."""
    if not edges:
        return
    conn = get_connection()
    try:
        conn.executemany(
            """INSERT OR IGNORE INTO graph_edges (id, source, target, relation, weight, document_id, chunk_id)
               VALUES (:id, :source, :target, :relation, :weight, :document_id, :chunk_id)""",
            edges,
        )
        conn.commit()
    finally:
        conn.close()


def get_all_graph_nodes(document_id: Optional[str] = None) -> list[dict]:
    """Fetch graph nodes, optionally filtered by document."""
    conn = get_connection()
    try:
        if document_id:
            rows = conn.execute(
                "SELECT * FROM graph_nodes WHERE document_id = ? ORDER BY name ASC",
                (document_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM graph_nodes ORDER BY name ASC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_graph_edges(document_id: Optional[str] = None) -> list[dict]:
    """Fetch graph edges, optionally filtered by document."""
    conn = get_connection()
    try:
        if document_id:
            rows = conn.execute(
                "SELECT * FROM graph_edges WHERE document_id = ? ORDER BY source ASC",
                (document_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM graph_edges ORDER BY source ASC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_graph_by_document(document_id: str) -> None:
    """Delete all graph nodes and edges for a specific document."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM graph_edges WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM graph_nodes WHERE document_id = ?", (document_id,))
        conn.commit()
    finally:
        conn.close()

