"""GRAG AI - FastAPI Application Entry Point."""

import structlog
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.api.ingestion import router as ingestion_router
from app.api.review_queue import router as review_queue_router
from app.api.openai import router as openai_router


settings = get_settings()
logger = structlog.get_logger()

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str
    environment: str
    services: Dict[str, Dict[str, str]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    logger.info(
        "app.startup",
        environment=settings.environment,
        log_level=settings.log_level,
        neo4j_uri=settings.neo4j_uri,
        ollama_url=settings.ollama_base_url,
    )

    from app.database.neo4j_client import get_neo4j_client
    from app.schemas.graph_schema import init_graph_schema

    neo4j_client = get_neo4j_client()
    connected = await neo4j_client.verify_connectivity()

    if connected:
        logger.info("app.neo4j.connected")
        schema_results = await init_graph_schema()
        logger.info("app.schema.initialized", results=schema_results)
    else:
        logger.warning("app.neo4j.not_connected")

    yield

    await neo4j_client.close()
    logger.info("app.shutdown")


app = FastAPI(
    title="GRAG AI",
    description="Hybrid Graph RAG System with dual-memory architecture",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingestion_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(review_queue_router, prefix="/api/v1", tags=["review-queue"])
app.include_router(openai_router, prefix="/v1", tags=["openai"])


@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check endpoint that verifies all service dependencies.

    Returns status of:
    - Neo4j connectivity
    - Ollama API availability
    - ChromaDB path accessibility
    """
    services = {
        "chromadb": {
            "status": "unknown",
            "accessible": False,
        },
        "ollama": {
            "status": "unknown",
            "available": False,
        },
        "neo4j": {
            "status": "unknown",
            "connected": False,
        },
    }

    overall_status = "healthy"

    try:
        chromadb_path = Path(settings.chromadb_path)
        if chromadb_path.exists() or chromadb_path.parent.exists():
            services["chromadb"]["status"] = "ok"
            services["chromadb"]["accessible"] = True
        else:
            services["chromadb"]["status"] = "warning"
            services["chromadb"]["accessible"] = False
    except Exception as e:
        services["chromadb"]["status"] = "error"
        services["chromadb"]["accessible"] = False
        overall_status = "degraded"
        logger.warning("health.chromadb.error", error=str(e))

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                services["ollama"]["status"] = "ok"
                services["ollama"]["available"] = True
            else:
                services["ollama"]["status"] = "error"
                overall_status = "degraded"
    except Exception as e:
        services["ollama"]["status"] = "unavailable"
        services["ollama"]["available"] = False
        overall_status = "degraded"
        logger.warning("health.ollama.error", error=str(e))

    try:
        from app.database.neo4j_client import get_neo4j_client

        neo4j_client = get_neo4j_client()
        connected = await neo4j_client.verify_connectivity()
        if connected:
            services["neo4j"]["status"] = "ok"
            services["neo4j"]["connected"] = True
        else:
            services["neo4j"]["status"] = "unavailable"
            overall_status = "degraded"
    except Exception as e:
        services["neo4j"]["status"] = "error"
        services["neo4j"]["connected"] = False
        overall_status = "degraded"
        logger.warning("health.neo4j.error", error=str(e))

    return HealthStatus(
        status=overall_status,
        environment=settings.environment,
        services=services,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "GRAG AI",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
