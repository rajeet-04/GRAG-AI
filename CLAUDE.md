<!-- GSD:project-start source:PROJECT.md -->
## Project

**GRAG AI — Hybrid Graph RAG System**

A dual-memory Graph RAG system featuring a temporally aware active knowledge graph (KR), user episodic memory (KB), and explainable AI (xAI) reasoning paths. Acts as a custom backend for OpenWebUI, providing traceable, transparent answers where knowledge evolves over time and the system shows *how it knows*, not just *what it knows*.

**Core Value:** Users receive explainable, traceable answers grounded in a living knowledge graph with temporal evolution — every answer comes with graph reasoning paths and confidence levels.

### Constraints

- **Hardware**: 8GB VRAM, 16GB RAM — all LLM operations must respect these limits
- **Neo4j Memory**: Heap strictly limited to 2GB via neo4j.conf
- **Token Limits**: Must implement tiktoken counting in Context Builder to prevent context-window crashes
- **Separation**: KR and KB must never mix — objective facts stay in Neo4j, subjective memory in vector store
- **API Compatibility**: Must expose OpenAI-compatible API for OpenWebUI integration
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **FastAPI** | 0.110+ | 5-Agent orchestration API | Async-first, native Pydantic support, OpenAPI auto-generation. Industry standard for AI agent APIs in 2026. |
| **LangGraph** | 0.2+ | Multi-agent workflow orchestration | Built on LangChain, native graph-based state management, handles agent handoffs, parallel execution, and tracing. Preferred over raw asyncio for complex agent graphs. |
| **Python** | 3.11+ | Runtime | Required by LangGraph, Neo4j drivers, Ollama SDK. 3.11+ for performance and async improvements. |
### Knowledge Representation (KR) — Neo4j
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Neo4j** | 5.20+ | Graph database with temporal versioning | Native graph traversal, Cypher query language, vector index support in DB. **neo4j-graphrag-python** official library provides GraphRAG primitives, LLM integrations, and retriever patterns. |
| **neo4j-graphrag-python** | 1.0+ | GraphRAG orchestration | Official Neo4j library with VectorRetriever, Text2Cypher, ToolsRetriever, and LLM abstraction layer compatible with Ollama. |
| **APOC** | 5.20+ | Neo4j procedures | Temporal versioning patterns, graph algorithms (community detection, PageRank for global search). |
### Knowledge Base (KB) — Vector Store
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **ChromaDB** | 0.5+ | Vector store for episodic/semantic memory | Embedded mode (no separate service), Python-native, handles 1M+ vectors comfortably on 4-8GB RAM. Simple operational footprint for local deployment. |
| **Alternative: Qdrant** | 1.10+ | If complex metadata filtering required | Superior pre-filtered vector search. Use if temporal queries need date range filters before similarity search. |
### Embeddings
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **nomic-embed-text** | 1.5 | Primary embedding model | 137M params, 768 dimensions, 8K context, 274MB. **Runs on CPU**, good retrieval quality, best-tested local embedding model. |
| **Qwen3-Embedding** | 0.6B | High-quality upgrade option | 0.6B params, 1024 dims, 32K context, 1.2GB. **70.7 MTEB v2** (dramatically higher quality). Slightly larger but still CPU-friendly. |
### LLM Runtime
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Ollama** | 0.5+ | Local LLM inference | OpenAI-compatible API, handles model downloads/quantization, works with 8GB VRAM for quantized 7B-14B models. Native OllamaLLM/OllamaEmbeddings in neo4j-graphrag-python. |
| **Recommended chat model:** Qwen2.5 7B Q4_K_M | Latest | Primary reasoning model | ~5GB VRAM quantized, excellent reasoning for code and general knowledge. Fits alongside embedding model in 8GB VRAM. |
| **Alternative:** Mistral 7B Q4_K_M | Latest | If Qwen unavailable | Similar VRAM profile, slightly different strengths. |
### Integration Layer
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **OpenWebUI** | 0.3+ | User-facing chat interface | RAG pipeline integration, Ollama backend, file upload, model management. Acts as frontend; your FastAPI handles the complex GRAG orchestration. |
| **MCP (Model Context Protocol)** | Latest | OpenWebUI ↔ FastAPI bridge | Expose GRAG as MCP server; OpenWebUI connects as MCP client. Enables tool-calling from chat interface. |
### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **httpx** | 0.27+ | Async HTTP client | Ollama API calls, MCP protocol |
| **pydantic** | 2.0+ | Data validation | Request/response schemas, structured output |
| **structlog** | 24.0+ | Structured logging | Agent tracing, debugging multi-agent flows |
| **tiktoken** | 0.7+ | Token counting | Cost metering, context window management |
| **tenacity** | 8.0+ | Retry logic | API resilience for Ollama calls |
## Alternatives Considered
| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Agent orchestration | LangGraph | Raw asyncio / CrewAI | LangGraph provides graph state, checkpointing, and built-in patterns for multi-agent handoffs. CrewAI is simpler but less flexible for custom graph topologies. |
| Vector store | ChromaDB | Qdrant | ChromaDB embedded mode requires no separate service—simpler for local deployment. Qdrant only if you need complex pre-filtered search. |
| Embedding model | nomic-embed-text | OpenAI embeddings | Local-only requirement. nomic-embed-text is free, private, CPU-friendly, and good enough for most retrieval tasks. |
| Graph DB | Neo4j | Memgraph | Neo4j has stronger ecosystem (neo4j-graphrag-python), Cypher is more widely known, and better tooling for temporal versioning patterns. |
| LLM API | Ollama (local) | OpenAI API | Local deployment requirement. Ollama handles quantization and serves OpenAI-compatible API. |
## Architecture Overview
## Installation
# Core dependencies
# Ollama models (run separately)
## Hardware Allocation (8GB VRAM, 16GB RAM)
| Component | VRAM | RAM | Notes |
|-----------|------|-----|-------|
| Qwen2.5 7B Q4_K_M | ~5GB | ~2GB | Main reasoning model |
| Ollama runtime | - | ~1GB | OS overhead |
| ChromaDB (embedded) | - | ~2-4GB | Vector storage, scales with corpus |
| Neo4j (local) | - | ~4-6GB | Graph + vector index |
| OS + misc | - | ~2-3GB | Baseline |
## Sources
- [Neo4j GraphRAG Python Documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html) — Official, HIGH confidence
- [The New Stack: Building Production AI Agents with FastAPI](https://thenewstack.io/how-to-build-production-ready-ai-agents-with-rag-and-fastapi/) — 2026-01-20, MEDIUM confidence
- [C# Corner: Multi-Agent Workflows for GraphRAG](https://www.c-sharpcorner.com/article/from-single-agent-to-multi-agent-workflows-three-orchestration-patterns-for-gra/) — 2026-03-04, MEDIUM confidence
- [4xxi: Vector Database Comparison 2026](https://4xxi.com/articles/vector-database-comparison) — 2026-03-18, HIGH confidence
- [InsiderLLM: Embedding Models for RAG](https://insiderllm.com/guides/embedding-models-rag/) — 2026-02-08, HIGH confidence
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) — Official, HIGH confidence
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
