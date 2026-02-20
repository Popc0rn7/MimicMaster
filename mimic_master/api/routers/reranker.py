"""Reranker API router."""

from fastapi import APIRouter

from mimic_master.models.reranker import RerankRequest, RerankResponse
from mimic_master.services.reranker_service import get_reranker_service

reranker_router = APIRouter()


@reranker_router.post("/", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """
    Rerank documents based on query relevance.

    Args:
        request: RerankRequest containing query, documents, and optional top_n

    Returns:
        RerankResponse with sorted indices and relevance scores
    """
    service = get_reranker_service()
    return await service.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
    )
