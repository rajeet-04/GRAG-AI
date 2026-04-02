# OpenWebUI Integration Guide

This guide explains how to connect OpenWebUI to GRAG AI as a custom backend.

## Prerequisites

- GRAG AI running on localhost:8000
- OpenWebUI running (localhost:3000 or deployed)
- Neo4j and Ollama services running

## Configuration

### 1. Start GRAG AI

```bash
cd R:\Code\GRAG AI
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Configure OpenWebUI

1. Open OpenWebUI in your browser
2. Navigate to **Settings** → **Models**
3. Add a new model with:
   - **Model Name**: `grag-pipeline-v1`
   - **API Endpoint**: `http://localhost:8000/v1`
   - **API Key**: (leave blank, or set API_KEY in GRAG's .env)

### 3. Verify Connection

- Select `grag-pipeline-v1` from the model dropdown
- Send a message
- You should receive responses with reasoning steps and Mermaid diagrams

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |

### Request Format

```json
{
  "model": "grag-pipeline-v1",
  "messages": [
    {"role": "user", "content": "Your question"}
  ],
  "stream": false
}
```

### Response Format

The response includes:
- `answer`: Natural language response
- `reasoning_steps`: Step-by-step graph reasoning
- `mermaid_path`: Mermaid.js diagram of graph traversal

## Troubleshooting

### Connection refused
- Ensure GRAG is running: `curl http://localhost:8000/health`
- Check port 8000 is not in use

### 401 Unauthorized
- Set API_KEY in GRAG's .env file
- Include in OpenWebUI settings

### Empty responses
- Check Neo4j has data
- Verify Ollama is running with required models
