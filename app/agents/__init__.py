"""
LangGraph agent nodes for GRAG AI.

Implements the 5-agent orchestration pattern:
1. Ingestion Agent — text → entities + relations
2. Graph Builder Agent — write to Neo4j with temporal versioning
3. Query Agent — natural language → Cypher
4. Context Builder Agent — merge KR + KB context
5. Explanation Agent — xAI reasoning output
"""

from app.agents.ingestion_agent import create_ingestion_state, ingestion_agent_node
from app.agents.explanation_agent import explanation_agent_node

__all__ = [
    "ingestion_agent_node",
    "create_ingestion_state",
    "explanation_agent_node",
]
