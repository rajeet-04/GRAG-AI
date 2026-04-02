# GRAG AI — Hybrid Graph RAG System

## Overview

GRAG AI is an experimental hybrid retrieval-augmented generation backend designed around a dual-memory architecture:
- Knowledge Representation (KR) in Neo4j (temporal graph, factual knowledge)
- Knowledge Base (KB) in ChromaDB (episodic + semantic user memory)
- Explainable AI (xAI) with traceable reasoning paths (graph traversal + Mermaid-friendly output)

It is built to integrate with OpenWebUI via an OpenAI-compatible HTTP API and uses Ollama for local LLM inference.

## Key Features

- FastAPI server with health checks and ingestion + review APIs
- Neo4j graph schema with temporal versioning and Graphiti support
- Ingestion pipeline: entity extraction, relationship extraction, document provenance
- Entity resolution with manual review queue and merge decision workflow
- Multi-agent architecture (planned): Ingestion, Graph Builder, Query, Context Builder, Explanation
- Retrieval engine (planned): graph-first multi-hop traversal with vector fallback
- Context window management via tiktoken token counting
- Strong separation of objective facts (KR) and subjective sessions (KB)

## Architecture

- `app/main.py`: FastAPI entrypoint, lifespan startup/shutdown, endpoints
- `app/config.py`: Pydantic settings (Neo4j, Ollama, ChromaDB, environment)
- `app/api/ingestion.py`: `/api/v1/ingest` endpoints
- `app/api/review_queue.py`: `/api/v1/review-queue` endpoints
- `app/ingestion`: entity/relation extraction + graph writer services
- `app/database`: Neo4j client and ChromaDB embedded client
- `app/memory`: short-term, episodic, semantic memory modules
- `app/retrieval`: fallback and ranking logic

## Requirements

### System
- Python 3.12+
- Neo4j 5.20+ with heap limited to 2GB (`neo4j/neo4j.conf`)
- Ollama local server (default: `http://localhost:11434`)
- 8GB VRAM, 16GB RAM

### Python dependencies
- FastAPI
- Uvicorn
- LangGraph
- pydantic, pydantic-settings
- Neo4j + neo4j-graphrag
- ChromaDB
- Ollama
- tiktoken, tenacity
- structlog, httpx

## Getting Started

1. Create `.env` with:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-q4_k_m
CHROMADB_PATH=./data/chromadb
LOG_LEVEL=INFO
ENVIRONMENT=development
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

3. Start required services:
- Start Neo4j (with `neo4j/neo4j.conf` 2GB heap)
- Start Ollama server and ensure model is available

4. Run server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Validate:
- `GET http://localhost:8000/health`
- `GET http://localhost:8000/docs`

## API Endpoints

- `GET /` : service metadata
- `GET /health` : health status for Neo4j, Ollama, ChromaDB sights
- `POST /api/v1/ingest` : ingest a single document
- `POST /api/v1/ingest/batch` : ingest multiple documents
- `GET /api/v1/review-queue` : list pending manual review decisions
- `GET /api/v1/review-queue/{decision_id}` : review detail
- `POST /api/v1/review-queue/{decision_id}/approve` : approve merge
- `POST /api/v1/review-queue/{decision_id}/reject` : reject merge

## Development Workflow

Project phases are defined in `.planning`:
- Phase 1: foundation & storage (complete)
- Phase 2: graph foundation (complete)
- Phase 3: KB memory architecture (complete)
- Phase 4: ingestion pipeline (complete)
- Phase 5: entity resolution (in progress)
- Phase 6+: core agents, retrieval engine, context management, explainability, OpenWebUI integration

## Testing

Run unit tests:

```bash
pytest -q
```

## Notes

- This repo is under active development; APIs and data models may change.
- Graph temporal invariants are enforced by `app/ingestion/graph_writer.py` and `.planning` requirements.
- Use the roadmaps and plan documents in `.planning` for implementation status and next tasks.

## Contributing

- Follow issue/PR template.
- Add features in a new phase under `.planning/phases`.
- Keep KR and KB separation strict in all code paths.

## License

TBD

