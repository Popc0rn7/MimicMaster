"""API routes for Mimic Master."""

from fastapi import APIRouter

from mimic_master.api.routers import (
    agent_router,
    embedding_router,
    reranker_router,
    retrieval_router,
    health_router,
)

# Main router
router = APIRouter()

# Include sub-routers
router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(embedding_router, prefix="/embeddings", tags=["embeddings"])
router.include_router(reranker_router, prefix="/reranker", tags=["reranker"])
router.include_router(retrieval_router, prefix="/retrieval", tags=["retrieval"])
router.include_router(agent_router, prefix="/agent", tags=["agent"])


def init_langsmith_tracing() -> None:
    """Initialize LangSmith tracing for the application."""
    try:
        from mimic_master.config import settings

        if settings.is_langsmith_configured:
            import os

            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
            if settings.langsmith_endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
            print(
                f"LangSmith tracing initialized for project: {settings.langsmith_project}"
            )
    except ImportError:
        print("LangChain not available for tracing")
