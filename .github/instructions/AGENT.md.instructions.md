---
description: Load when a user request touches AGENT implementation, Graphiti/OLLAMA integration, or in-repo agent instructions.
applyTo: "AGENT/**"
---

# AGENT Guidance for Copilot

Use these instructions when working on anything related to the `AGENT` documentation or code paths in this workspace.

## When to apply these instructions

- User asks for new or updated agent docs in `AGENT/GRAPHITI.md`, `AGENT/GRAPHITI_SDK.md`, `AGENT/OLLAMA.md`
- User asks for best practices for local Ollama model usage, GraphRAG, or hybrid Graph+vector retrieval architecture in this repo
- User asks for instructions referencing OpenWebUI/MCP interop as described by this project’s stack
- User asks to generate new agent definitions, prompt templates, or to fix behavior in cross-agent workflows

## Project context (from AGENT docs)

- `AGENT/GRAPHITI.md` is a Graphiti quick-start guide (Neo4j+LLM ingestion/search, environment variables, concurrency, graph search, center node reranking, hybrid search). Follow these assumptions for feature requests and docs.
- `AGENT/GRAPHITI_SDK.md` contains API schema examples for thread management in Zep/Graphiti SDK style (GET/POST endpoints, request/response models). Leverage this style for endpoint contract docs.
- `AGENT/OLLAMA.md` is Ollama user guidance (local/cloud model runtime, API configuration, Python/JS/cURL examples, streaming behavior, tool calling). Use this as canonical guidance for Ollama integration.

## Coding guidelines

- Keep code and docs concise, with explicit environment setup steps (`NEO4J_URI`, `OPENAI_API_KEY`, `OLLAMA_API_KEY`, `SEMAPHORE_LIMIT` as in docs).
- Prefer idiomatic FastAPI/Pydantic patterns for API layers (in app/api and app/services), aligned to this project’s existing style.
- Maintain strict separation of knowledge stores: objective facts in `app/database/neo4j_client.py`, episodic memory in `app/database/chroma_client.py`.
- Apply tiktoken-style token counting in context builder code paths (existing requirement in CLAUDE.md).
- Keep new features compatible with local 8GB VRAM, 16GB RAM constraint (use quantized Ollama models, small embedders like `nomic-embed-text`).

## Review checklist for agent documentation/code changes

- Verify new instructions mention supported model providers and local/offline mode (Ollama local/cloud) exactly as in `AGENT/OLLAMA.md`.
- Verify Neo4j/Graphiti setup instructions mention connection and credentials and that the data model uses temporal validity (graph facts with valid_at/invalid_at).
- Validate that API docs follow existing style and include OpenAPI-contract-like properties, status codes, and sample payloads.
- Avoid introducing non-free/unsupported external services no longer in stack (remote OpenAI unless explicitly asked and permitted).

> Note: this is the recommended instructions file state for AI agents operating in this repository. Keep it synced with the actual AGENT/ Markdown content.
