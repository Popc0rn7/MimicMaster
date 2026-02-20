"""Retrieval request and response models."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """A retrieved document from the vector database."""

    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Similarity score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class RetrievalRequest(BaseModel):
    """Request model for retrieval."""

    query: str = Field(..., description="Query text")
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    filter: Optional[dict] = Field(default=None, description="Metadata filter")
    namespace: str = Field(default="", description="Pinecone namespace")


class RetrievalResponse(BaseModel):
    """Response model for retrieval results."""

    results: List[RetrievedDocument] = Field(..., description="Retrieved documents")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")
