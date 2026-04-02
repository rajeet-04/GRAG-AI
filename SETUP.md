# GRAG AI Setup Guide

## Prerequisites

### Hardware Requirements
- **VRAM**: 8GB (for LLM models)
- **RAM**: 16GB
- **Disk**: 10GB+ for Neo4j, ChromaDB, models

### Software Requirements
- Python 3.12+
- Docker (optional but recommended)
- Neo4j 5.20+
- Ollama

## Option 1: Docker Setup (Recommended)

### Step 1: Configure Environment

```bash
# Clone repository
git clone https://github.com/yourusername/grag-ai.git
cd grag-ai

# Copy environment template
cp env.example .env
```

Edit `.env` with your settings:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:7b-q4_k_m
CHROMADB_PATH=/data/chromadb
```

### Step 2: Start Neo4j

```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.20
```

Or download and run Neo4j Desktop.

### Step 3: Build and Run GRAG

```bash
# Build Docker image
docker build -t grag-ai .

# Run container
docker run -d \
  --name grag-ai \
  -p 8000:8000 \
  --env-file .env \
  grag-ai
```

## Option 2: Local Setup

### Step 1: Install Python Dependencies

```bash
# Using uv (recommended)
cd grag-ai
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp env.example .env
# Edit .env with your settings
```

### Step 3: Start Required Services

#### Neo4j

**Option A: Docker**
```bash
docker run -d \
  --name neo4j \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.20
```

**Option B: Neo4j Desktop**
1. Download Neo4j Desktop
2. Create new database
3. Set password to match your `.env`

**Configure 2GB Heap Limit:**
In `neo4j.conf`:
```properties
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=2g
```

#### Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# Pull required models
ollama pull qwen2.5:7b-q4_k_m
ollama pull nomic-embed-text
```

### Step 4: Run GRAG

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## OpenWebUI Integration

### Step 1: Start GRAG

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 2: Configure OpenWebUI

1. Open OpenWebUI in browser
2. Go to **Admin Panel** → **Settings** → **AI Settings** → **Advanced**
3. Add new API connection:
   - **API Type**: OpenAI Compatible
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: (leave blank, or set in `.env`)
4. Select model: `grag-pipeline-v1`

### Step 3: Test Integration

```bash
# Health check
curl http://localhost:8000/health

# Test chat completions
curl http://localhost:8000/v1/models

# Test with streaming
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "grag-pipeline-v1", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Test connection
curl http://localhost:7474
```

### Ollama Connection Issues

```bash
# Check Ollama is running
curl http://localhost:11434

# Pull model again
ollama pull qwen2.5:7b-q4_k_m
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
PORT=8001 uvicorn app.main:app --port 8001
```

### Memory Issues

If you encounter OOM errors:
1. Use smaller LLM models (qwen2.5:3b-q4_k_m instead of 7b)
2. Reduce ChromaDB memory in config
3. Ensure Neo4j heap is limited to 2GB

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| NEO4J_URI | Yes | bolt://localhost:7687 | Neo4j connection URI |
| NEO4J_USER | Yes | neo4j | Neo4j username |
| NEO4J_PASSWORD | Yes | password | Neo4j password |
| OLLAMA_BASE_URL | Yes | http://localhost:11434 | Ollama API URL |
| OLLAMA_MODEL | Yes | qwen2.5:7b-q4_k_m | Chat model |
| EMBEDDING_MODEL | Yes | nomic-embed-text | Embedding model |
| CHROMADB_PATH | Yes | ./data/chromadb | Vector store path |
| API_KEY | No | (empty) | OpenWebUI API key |
| MODEL_NAME | No | grag-pipeline-v1 | OpenWebUI model name |
| CONTEXT_TOKEN_BUDGET | No | 8192 | Max tokens in context |
| LOG_LEVEL | No | INFO | Logging level |

## Docker Compose (Alternative)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  grag:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/data
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:5.20
    ports:
      - "7687:7687"
      - "7474:7474"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

Run with:
```bash
docker-compose up -d
```

## Next Steps

- Review `README.md` for API documentation
- Check `docs/OPENWEBUI_INTEGRATION.md` for detailed integration steps
- Explore `.planning/` directory for implementation details
