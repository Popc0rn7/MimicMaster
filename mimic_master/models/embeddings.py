"""Embedding request and response models."""

from typing import List

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""

    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)


class EmbeddingResponse(BaseModel):
    """Response model for text embeddings."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Dimension of each embedding vector")
