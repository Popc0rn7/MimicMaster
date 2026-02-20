"""Embedding request and response models."""

from typing import List, Dict

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""

    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)


class SparseVector(BaseModel):
    """Sparse vector representation."""

    indices: List[int] = Field(..., description="Non-zero indices")
    values: List[float] = Field(..., description="Values at those indices")


class DenseAndSparseEmbeddings(BaseModel):
    """Dense and sparse embeddings for a single text."""

    dense: List[float] = Field(..., description="Dense embedding vector")
    sparse: SparseVector = Field(..., description="Sparse embedding vector")


class EmbeddingResponse(BaseModel):
    """Response model for text embeddings (dense + sparse)."""

    embeddings: List[DenseAndSparseEmbeddings] = Field(
        ..., description="List of dense and sparse embeddings"
    )
    dense_dimension: int = Field(..., description="Dimension of dense vectors")


class DenseEmbeddingResponse(BaseModel):
    """Response model for dense-only embeddings (backward compatibility)."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Dimension of each embedding vector")
