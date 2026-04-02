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

### Option 1: Docker Compose Setup (Recommended)

The full stack (Neo4j, Ollama, GRAG API, and Open WebUI) is managed via a single `docker-compose.yml`.

### Step 1: Install NVIDIA Container Toolkit
If you have an NVIDIA GPU, ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on your host to enable GPU acceleration for Ollama.

### Step 2: Configure Environment

```bash
# Clone repository
git clone https://github.com/rajeet-04/GRAG-AI.git
cd grag-ai

# Create your .env
cp env.example .env
```

### Step 3: Configure Hybrid Cloud (Optional but Recommended)
For the best performance, enable **Ollama Cloud** for reasoning tasks while keeping ingestion local:
1. Get your API key from [ollama.com](https://ollama.com).
2. Edit `.env`:
   ```env
   OLLAMA_CLOUD_ENABLED=true
   OLLAMA_CLOUD_API_KEY=your_key_here
   OLLAMA_CLOUD_MODEL=minimax-m2.7:cloud
   ```

### Step 4: Start the Stack

```bash
# Pull and start all services
docker compose up -d

# Monitor model pull progress (required on first run)
docker logs -f grag-ollama-init
```

### Step 5: Access the UI
- **Open WebUI**: [http://localhost:3000](http://localhost:3000)
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474)
- **GRAG API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Option 2: Manual / Local Setup

### Step 1: Install Dependencies
```bash
# Using uv
uv sync

# Or pip
pip install -r requirements.txt
```

### Step 2: Start Neo4j & Ollama
Ensure Neo4j (bolt://localhost:7687) and Ollama (http://localhost:11434) are running on your host.

### Step 3: Run the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Troubleshooting

### ⚡ Latency & Streaming
**Issue**: Message takes 10+ seconds to start typing.
**Solution**: This is normal for the first-token latency (~3-5s for graph search). Ensure `stream: true` is set in your client. GRAG AI uses a two-phase stream: the "Processing" message arrives via segment 1, then the LLM tokens arrive via native SSE.

### 🧩 Mermaid Diagram Parse Errors
**Issue**: Diagrams show a red "Parse Error" in Open WebUI.
**Solution**: This usually happens if the LLM generates complex labels with quotes. We have added strict system prompt rules to prevent this. If it persists, check `app/agents/explanation_agent.py` for the Mermaid formatting rules.

### 🐳 Docker GPU Issues
**Issue**: `could not select device driver "" with capabilities: [gpu]`
**Solution**: You don't have the NVIDIA Container Toolkit installed or `nvidia-container-runtime` isn't the default. You can remove the `deploy: resources` section from `docker-compose.yml` to run on CPU only (very slow).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_CLOUD_ENABLED` | `false` | Enable/Disable cloud offloading |
| `OLLAMA_CLOUD_API_KEY` | - | Required if cloud is enabled |
| `NEO4J_URI` | `bolt://neo4j:7687` | Use `localhost` for local, `neo4j` for docker |
| `CONTEXT_TOKEN_BUDGET` | `8192` | Adjust based on your model's capacity |
| `API_KEY` | - | Required for Open WebUI bearer auth |

---

## Next Steps
- Review `README.md` for architecture details.
- Review `AGENT/OLLAMA.md` for protocol specifications.
