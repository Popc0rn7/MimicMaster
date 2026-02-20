"""FastAPI application factory and main app."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mimic_master.config import settings
from mimic_master.api.routes import (
    router,
    init_langsmith_tracing,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Initialize LangSmith tracing
    if settings.is_langsmith_configured:
        init_langsmith_tracing()
        print(f"LangSmith tracing enabled for project: {settings.langsmith_project}")

    # Check service configurations
    if settings.is_pinecone_configured:
        print(f"Pinecone configured with index: {settings.pinecone_index}")
    else:
        print("Pinecone not configured. Set PINECONE_API_KEY and PINECONE_INDEX in .env")

    if settings.use_mock_embedding:
        print("Using mock embedding service")
    else:
        print(f"Using embedding service at: {settings.embedding_provider_url}")

    if settings.use_mock_reranker:
        print("Using mock reranker service")
    else:
        print(f"Using reranker service at: {settings.reranker_provider_url}")

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Mimic Master",
        description="D&D 5E Dungeon Master AI Agent Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router, prefix="/api/v1")

    return app


# Global app instance for direct import
app = create_app()
