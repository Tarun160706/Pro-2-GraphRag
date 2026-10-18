"""
Structured logging configuration.

Sets up a consistent log format across the application.
Log level is controlled by APP_DEBUG in the environment:
  True  → DEBUG
  False → INFO
"""

import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    """Configure the root logger for the application."""
    log_level = logging.DEBUG if settings.app_debug else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called more than once
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
