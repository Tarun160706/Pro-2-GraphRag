# GraphRAG Knowledge Intelligence Assistant

A hybrid **Graph + RAG** knowledge assistant that ingests documents, builds a knowledge graph, creates vector embeddings, and combines both retrieval methods to generate grounded answers with source references.

## Architecture

```
Documents
    │
    ▼
Document Processing (PDF / DOCX / TXT)
    │
    ▼
Text Chunking
    │
    ├──────────────────────────┐
    ▼                          ▼
Embeddings               Entity / Relation
(Sentence Transformers)   Extraction
    │                          │
    ▼                          ▼
ChromaDB                  NetworkX
(Vector Store)            (Knowledge Graph)
    │                          │
    └───────────┬──────────────┘
                ▼
        Hybrid Retrieval
                │
                ▼
        LLM Generation
                │
                ▼
     Grounded Answer + Sources
```

## Tech Stack

| Layer        | Technology                              |
| ------------ | --------------------------------------- |
| API          | FastAPI, Uvicorn                        |
| Embeddings   | Sentence Transformers (all-MiniLM-L6-v2)|
| Vector DB    | ChromaDB (local persistent)             |
| Knowledge Graph | NetworkX                             |
| Metadata DB  | SQLite                                  |
| LLM          | Pluggable (free-tier external API)      |
| Frontend     | HTML5, Vanilla CSS (Glassmorphic Design), Vis.js Network, JavaScript |

## Development Roadmap

- [x] **Stage 1 — Foundation**: Project structure, FastAPI app skeleton, SQLite database schema, settings, health endpoint
- [x] **Stage 2 — Document Ingestion**: PDF/DOCX/TXT/MD loaders, recursive text chunking, SQLite metadata
- [x] **Stage 3 — Embeddings & Vector Store**: Sentence Transformers (all-MiniLM-L6-v2), ChromaDB persistent vector store
- [x] **Stage 4 — Knowledge Graph**: NER & relation extraction, NetworkX multigraph, visual subgraph endpoints
- [x] **Stage 5 — Hybrid Retrieval & Generation**: Tri-mode retrieval (Hybrid RRF, Vector, Graph), Gemini API + Local Grounded Synthesizer
- [x] **Stage 6 — Interactive Frontend Dashboard**: Modern glassmorphic SPA, Vis.js force-directed graph visualizer, drag-and-drop ingestion, intelligent chat assistant
- [ ] **Stage 7 — Production Polish & Evaluation**: End-to-end benchmarking, latency profiling, deployment packaging

## Quick Start

### Prerequisites

- Python 3.11+
- Windows (developed and tested on Windows with 8 GB RAM)

### Setup

```bash
# Navigate to the project root
cd graphrag-knowledge-assistant

# Create a virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt

# Copy environment config
copy .env.example .env
```

### Run the GraphRAG Application & Dashboard

```bash
cd backend
python -m uvicorn app.main:app --reload
```

The interactive web dashboard and API will be immediately available at:
- **Interactive Web App**: **http://127.0.0.1:8000/**
- **Health check**: `GET /health`
- **Interactive Swagger docs**: `GET /docs`

### Configuration

All configuration is via environment variables (or `.env` file). Key settings:

| Variable         | Default           | Description                        |
| ---------------- | ----------------- | ---------------------------------- |
| `APP_ENV`        | `development`     | Environment name                   |
| `APP_DEBUG`      | `true`            | Enable debug logging               |
| `DATA_DIR`       | `./data`          | Root data directory                 |
| `DOCUMENTS_DIR`  | `./data/documents`| Uploaded documents                  |
| `CHROMA_DIR`     | `./data/chroma`   | ChromaDB persistent storage        |
| `GRAPH_DIR`      | `./data/graph`    | NetworkX graph files                |
| `SQLITE_DB_PATH` | `./data/graphrag.db` | SQLite database path             |

To store data on a different drive, set absolute paths in `.env`:

```env
DATA_DIR=E:\graphrag-data
DOCUMENTS_DIR=E:\graphrag-data\documents
CHROMA_DIR=E:\graphrag-data\chroma
GRAPH_DIR=E:\graphrag-data\graph
PROCESSED_DIR=E:\graphrag-data\processed
SQLITE_DB_PATH=E:\graphrag-data\graphrag.db
```

## Project Structure

```
graphrag-knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── api/                 # Route handlers
│   │   │   └── health.py        # Health-check endpoint
│   │   ├── core/                # Configuration & logging
│   │   │   ├── config.py        # Settings from environment
│   │   │   └── logging_config.py
│   │   ├── ingestion/           # Document loaders & chunking (Stage 2)
│   │   ├── embeddings/          # Embedding service (Stage 3)
│   │   ├── vector/              # ChromaDB store (Stage 3)
│   │   ├── graph/               # Knowledge graph (Stage 4)
│   │   ├── retrieval/           # Retrieval strategies (Stage 5)
│   │   ├── llm/                 # LLM abstraction (Stage 6)
│   │   ├── database/            # SQLite models (Stage 2)
│   │   └── schemas/             # Pydantic schemas
│   ├── tests/
│   └── requirements.txt
├── frontend/                    # React app (Stage 7)
├── data/                        # Runtime data (git-ignored)
│   ├── documents/
│   ├── chroma/
│   ├── graph/
│   └── processed/
├── scripts/                     # Utility scripts
├── .env.example
├── .gitignore
└── README.md
```

## License

This project is for educational and portfolio purposes.
