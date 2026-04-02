# GRAG AI - Hybrid Graph RAG System

## Overview

GRAG AI is a production-ready hybrid retrieval-augmented generation backend with a dual-memory architecture:
- **Knowledge Representation (KR)**: Neo4j temporal graph for factual knowledge
- **Knowledge Base (KB)**: ChromaDB for episodic + semantic user memory
- **Explainable AI (xAI)**: Traceable reasoning paths with Mermaid visualization

Integrates with OpenWebUI via OpenAI-compatible API and uses Ollama for local LLM inference.

## Key Features

- ✅ Dual-memory architecture (KR/KB strict firewall)
- ✅ Hybrid Inference: Local Ingestion + Ollama Cloud reasoning for speed
- ✅ Temporal versioning in Neo4j (valid_from/valid_to)
- ✅ 6-agent LangGraph orchestration (Ingestion, Graph Builder, Query, Parallel Search, Context, Explanation)
- ✅ Multi-hop graph traversal (2-4 hops) with vector fallback
- ✅ Context window management with tiktoken (8192 token budget)
- ✅ Explainable AI (xAI) with reasoning paths + Mermaid diagrams
- ✅ True Streaming support (SSE) with real-time token delivery
- ✅ OpenAI-compatible API (`/v1/chat/completions`) for Open WebUI integration

## Architecture

```
app/
├── main.py              # FastAPI entrypoint
├── config.py           # Pydantic settings
├── api/
│   ├── ingestion.py    # Document ingestion endpoints
│   ├── openai.py       # OpenAI-compatible API
│   └── review_queue.py # Manual review endpoints
├── agents/
│   ├── graph.py        # LangGraph orchestration
│   ├── query_agent.py  # NL → Cypher
│   ├── context_builder.py # KR + KB merging
│   └── explanation_agent.py # xAI output
├── database/
│   ├── neo4j_client.py # Neo4j operations
│   └── chroma_client.py # ChromaDB vector store
├── memory/             # KB: short-term, episodic, semantic
├── retrieval/          # Fallback + ranking
└── schemas/            # Pydantic models
```

## Quick Start

### Prerequisites

- Python 3.12+
- Neo4j 5.20+ (with 2GB heap limit)
- Ollama with models
- 8GB VRAM, 16GB RAM recommended

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/rajeet-04/GRAG-AI.git
cd grag-ai

# Create your .env
cp env.example .env
```

### 2. Configure Environment

Edit `.env` with your settings:

```env
# Required: Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Ollama Cloud Configuration (Context Builder, Explanation Agent — High Performance)
OLLAMA_CLOUD_URL=https://ollama.com
OLLAMA_CLOUD_MODEL=minimax-m2.7:cloud
OLLAMA_CLOUD_API_KEY=your_cloud_api_key_here
OLLAMA_CLOUD_ENABLED=true  # Set to true to offload reasoning to cloud

# Required: ChromaDB
CHROMADB_PATH=./data/chromadb

# Optional: API Authentication (blank = no auth)
API_KEY=

# Optional: Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Step 3: Deployment with Docker Compose (GPU & Hybrid)

The recommended way to run GRAG AI is via the included `docker-compose.yml`, which handles Neo4j, local Ollama (with GPU support), the GRAG API, and Open WebUI automatically.

### Step 4: Start the Stack

```bash
# Pull and start all services
docker compose up -d

# Monitor model pull progress (required on first run)
docker logs -f grag-ollama-init
```

**Key Docker Services:**
- **`grag-api`**: The FastAPI backend (port 8000)
- **`grag-open-webui`**: The chat UI (port 3000)
- **`grag-ollama`**: Local inference with NVIDIA GPU passthrough
- **`grag-neo4j`**: Knowledge Graph storage

### Step 4: Verify Models

Wait for the `grag-ollama-init` service to finish pulling models:
```bash
docker logs -f grag-ollama-init
```

### 4. Run Locally

```bash
# Using uv (recommended)
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# Or using pip
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## OpenWebUI Integration

### Configuration

1. **Start GRAG AI**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **Configure OpenWebUI**:
   - Go to Admin Panel → Settings → AI Settings → Advanced
   - Add new API connection:
     - **API Base URL**: `http://localhost:8000/v1`
     - **API Key**: (leave blank if not set in .env)
   - Select the model: `grag-pipeline-v1`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/api/v1/ingest` | POST | Ingest documents |
| `/api/v1/review-queue` | GET | List pending reviews |

### True Streaming

GRAG AI implements a two-phase streaming architecture for maximum responsiveness:
1. **Pipeline Processing**: Runs agentic search + context building (~1-3s).
2. **Token Streaming**: Once context is ready, tokens stream in real-time from Ollama Cloud/Local to the UI.

This ensures you see `[GRAG is processing your request...]` immediately, followed by token-by-token generation once the facts are retrieved.

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "grag-pipeline-v1", "messages": [{"role": "user", "content": "Explain binary search"}], "stream": true}'
```

## Development

### Running Tests

```bash
pytest -q
```

### Project Phases

All 10 phases complete (v1.0 milestone):

1. Foundation & Storage
2. Graph Foundation  
3. KB Memory Architecture
4. Ingestion Pipeline
5. Entity Resolution
6. Core Agents
7. Retrieval Engine
8. Context Management
9. Explainability
10. OpenWebUI Integration

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| VRAM | 8GB (for LLM) |
| RAM | 16GB |
| Neo4j Heap | 2GB (configured in neo4j.conf) |

## License

GNU General Public License v3

## Support

- Open an issue on GitHub
- Check docs in `docs/` directory
- Review `.planning/` for implementation details
