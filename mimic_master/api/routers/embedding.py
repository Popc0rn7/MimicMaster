"""Embedding API router."""

from fastapi import APIRouter

from mimic_master.models.embeddings import EmbeddingRequest, EmbeddingResponse
from mimic_master.services.embedding_service import get_embedding_service

embedding_router = APIRouter()


@embedding_router.post("/", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for the given texts.

    Args:
        request: EmbeddingRequest containing list of texts

    Returns:
        EmbeddingResponse with embedding vectors
    """
    service = get_embedding_service()
    return await service.embed(request.texts)
