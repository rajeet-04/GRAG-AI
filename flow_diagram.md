# GRAG AI: Architecture & Data Flow

This document outlines the dual-memory architecture of GRAG AI. It is broken down into two primary lifecycles: Data Ingestion (Storage) and Query Resolution (Inference).

---

## 1. Data Ingestion & Storage Pipeline

The ingestion pipeline is responsible for safely parsing new knowledge, extracting structured graph entities, and maintaining the strict firewall between objective facts (KR) and semantic chunks (KB).

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    Doc[Document / Text Input]
    API[FastAPI Ingest Endpoint\n/api/v1/ingest]
    Orchestrator[LangGraph Ingestion Agent]
    
    Chunker[Text Chunker & Cleaner]
    LLMExtract[LLM Entity & Relation Extractor\nOllama / Qwen2.5]
    Embedder[Embedding Model\nnomic-embed-text]
    
    Neo4j[(Neo4j KR\nTemporal Graph)]
    Chroma[(ChromaDB KB\nVector Store)]

    Doc -->|POST Request| API
    API --> Orchestrator
    Orchestrator --> Chunker
    
    Chunker -->|Raw Text| LLMExtract
    LLMExtract -->|Extract Nodes & Edges| GraphBuilder[Graph Builder Agent]
    GraphBuilder -->|Cypher Merge| Neo4j
    
    Chunker -->|Text Chunks| Embedder
    Embedder -->|768-dim Vectors| ChromaDBStore[Vector Indexing]
    ChromaDBStore --> Chroma
```

---

### Ingestion Breakdown

* Input: Raw documents hit FastAPI
* Orchestration: LangGraph manages flow
* Graph Path (KR): Extract entities + relationships → Neo4j
* Vector Path (KB): Chunk + embed → ChromaDB

---

## 2. Query Resolution & Generation Pipeline

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    User[User Query\nvia OpenWebUI]
    UI[OpenWebUI Interface]
    
    API[FastAPI Endpoint\n/v1/chat/completions]
    LangGraph[LangGraph Orchestrator]
    
    QueryAgent[Query Agent\nNL to Cypher]
    SearchAgent[Search Agent]
    ContextAgent[Context Builder]
    ExplanationAgent[xAI Agent]
    
    Neo4j[(Neo4j KR)]
    Chroma[(ChromaDB KB)]
    
    Ollama[Local LLM\nQwen2.5]

    User --> UI
    UI --> API
    API --> LangGraph
    
    LangGraph --> QueryAgent
    QueryAgent --> Neo4j
    
    Neo4j --> ContextAgent
    Neo4j -.-> SearchAgent
    
    SearchAgent --> Chroma
    Chroma --> ContextAgent
    
    ContextAgent --> Merged[Merged Context]
    Merged --> ExplanationAgent
    ExplanationAgent --> Prompt[Prompt Builder]
    Prompt --> Ollama
    
    Ollama --> API
    API --> UI
```

---

### Query Breakdown

* Graph-first retrieval (Neo4j)
* Fallback to semantic search (ChromaDB)
* Context trimming (token safety)
* Explainability via reasoning path
* Streaming output via SSE

---

## 3. Real-Time Streaming (SSE)

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    Ollama[Local LLM]
    API[FastAPI SSE]
    UI[OpenWebUI]
    User((User))

    Ollama --> API
    API --> UI
    UI --> User
```

---

### Streaming Flow

* LLM generates tokens incrementally
* FastAPI streams via SSE
* UI renders instantly (real-time typing effect)

---

## Final Architecture Summary

* **KR (Neo4j):** structured reasoning
* **KB (Chroma):** semantic fallback
* **LangGraph:** orchestration
* **Ollama:** local inference
* **SSE:** real-time UX

---

## Done

This now:

* ✅ Works in VS Code preview
* ✅ Works on GitHub
* ✅ No rendering errors
* ✅ Clean + readable
