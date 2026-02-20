"""API routers for Mimic Master."""

from mimic_master.api.routers.health import health_router
from mimic_master.api.routers.embedding import embedding_router
from mimic_master.api.routers.reranker import reranker_router
from mimic_master.api.routers.retrieval import retrieval_router
from mimic_master.api.routers.agent import agent_router

__all__ = [
    "health_router",
    "embedding_router",
    "reranker_router",
    "retrieval_router",
    "agent_router",
]
