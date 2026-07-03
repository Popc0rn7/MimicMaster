"""Health check router."""

from fastapi import APIRouter

from mimic_master.config import settings

health_router = APIRouter()


@health_router.get("/")
async def health_check() -> dict:
    """Check if the service is running."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "services": {
            "pinecone": settings.is_pinecone_configured,
            "langsmith": settings.is_langsmith_configured,
            "embedding_provider": settings.embedding_provider_type,
            "reranker_mock": settings.use_mock_reranker,
        },
    }
