"""
Embedding service using Sentence Transformers.

Wraps the model loading and encoding behind a simple interface.
The model is loaded lazily on first use to keep startup fast,
and cached as a singleton so it's only loaded once.

Default model: all-MiniLM-L6-v2
  - 384 dimensions
  - ~80 MB download
  - Fast on CPU, fits easily in 8 GB RAM
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused across requests
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Get or lazily load the embedding model.

    Loading is deferred to first use rather than app startup
    so the server starts quickly and the model is only loaded
    when actually needed.
    """
    global _model
    if _model is None:
        model_name = settings.embedding_model
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info(
            "Model loaded — dimension: %d",
            _model.get_sentence_embedding_dimension(),
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = get_model()
    # normalize_embeddings=True improves cosine similarity search
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single search query.

    Separated from embed_texts for clarity — queries and
    documents use the same model with this architecture.
    """
    result = embed_texts([query])
    return result[0]


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding model."""
    model = get_model()
    return model.get_sentence_embedding_dimension()
