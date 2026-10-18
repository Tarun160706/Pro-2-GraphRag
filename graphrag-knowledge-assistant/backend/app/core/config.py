"""
Application configuration loaded from environment variables.

Uses pydantic-settings so every value is typed, validated, and
documented in one place. Defaults point to ./data/ relative to
the project root; override them in .env to use a different drive.
"""

import os
from pathlib import Path
from typing import Optional
import certifi
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure requests/urllib always find a valid CA cert bundle even if Windows registry/env is misconfigured
_ca_bundle = certifi.where()
os.environ.setdefault("SSL_CERT_FILE", _ca_bundle)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_bundle)
# If existing env vars point to non-existent files, overwrite them with certifi bundle
for _env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
    _val = os.environ.get(_env_var)
    if _val and not os.path.exists(_val):
        os.environ[_env_var] = _ca_bundle


# Resolve project root: two levels up from this file
# (backend/app/core/config.py → backend/app/core → backend/app → backend → project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Central configuration sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "GraphRAG Knowledge Intelligence Assistant"
    app_env: str = "development"
    app_debug: bool = True

    # ── Server ───────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000

    # ── Data storage paths ───────────────────────────────────
    # Paths are strings so users can set absolute paths like
    # E:\graphrag-data\documents in their .env file.
    data_dir: str = str(_PROJECT_ROOT / "data")
    documents_dir: str = str(_PROJECT_ROOT / "data" / "documents")
    chroma_dir: str = str(_PROJECT_ROOT / "data" / "chroma")
    graph_dir: str = str(_PROJECT_ROOT / "data" / "graph")
    processed_dir: str = str(_PROJECT_ROOT / "data" / "processed")
    sqlite_db_path: str = str(_PROJECT_ROOT / "data" / "graphrag.db")
    frontend_dir: str = str(_PROJECT_ROOT / "frontend")

    # ── Embeddings ───────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    # ChromaDB collection name for document chunks
    chroma_collection: str = "document_chunks"

    # ── LLM Configuration ────────────────────────────────────
    gemini_api_key: Optional[str] = None
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.2

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        """Anchor any relative data paths to the project root directory."""
        for field in [
            "data_dir",
            "documents_dir",
            "chroma_dir",
            "graph_dir",
            "processed_dir",
            "sqlite_db_path",
            "frontend_dir",
        ]:
            val = getattr(self, field, None)
            if val:
                p = Path(val)
                if not p.is_absolute():
                    setattr(self, field, str((_PROJECT_ROOT / p).resolve()))
        return self

    def ensure_data_dirs(self) -> None:
        """Create all configured data directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.documents_dir,
            self.chroma_dir,
            self.graph_dir,
            self.processed_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Singleton instance used throughout the app
settings = Settings()
