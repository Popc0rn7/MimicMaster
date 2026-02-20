"""Reranker request and response models."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """Request model for reranking documents."""

    query: str = Field(..., description="Query text")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_n: Optional[int] = Field(
        default=None, description="Number of top results to return", ge=1
    )


class RerankResponse(BaseModel):
    """Response model for reranking results."""

    results: List[int] = Field(
        ..., description="Indices of documents sorted by relevance score"
    )
    scores: List[float] = Field(..., description="Relevance scores for each result")
