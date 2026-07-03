"""Embedding service with NVIDIA and self-hosted HTTP provider support."""

from __future__ import annotations

import httpx
from typing import TYPE_CHECKING, Dict, List, Optional
from openai import OpenAI
from openai.types import Embedding

from mimic_master.config import settings
from mimic_master.models.embeddings import (
    EmbeddingResponse,
    DenseAndSparseEmbeddings,
    SparseVector,
)

if TYPE_CHECKING:
    from pinecone import Pinecone


class EmbeddingService:
    """Service for generating text embeddings using BGE-M3 model."""

    def __init__(self) -> None:
        self._dimension: int = settings.embedding_dimension
        self._nvidia_client: Optional["OpenAI"] = None
        self._pinecone_client: Optional["Pinecone"] = None

    def _get_nvidia_client(self) -> "OpenAI":
        """Get or create OpenAI-compatible client for NVIDIA NIM."""
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA embedding is configured but NVIDIA_API_KEY is not set."
            )
        if self._nvidia_client is None:
            self._nvidia_client = OpenAI(
                api_key=settings.nvidia_api_key,
                base_url=settings.nvidia_base_url,
            )
        return self._nvidia_client

    def _get_pinecone_client(self) -> "Pinecone":
        """Get or create Pinecone client for sparse embeddings."""
        if self._pinecone_client is None:
            from pinecone import Pinecone

            self._pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
        return self._pinecone_client

    async def embed(self, texts: List[str]) -> EmbeddingResponse:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResponse containing dense and sparse embeddings

        Raises:
            httpx.HTTPError: If the external service fails
        """
        if settings.embedding_provider_type == "nvidia":
            return await self._nvidia_embed(texts)

        if settings.use_http_embedding:
            return await self._http_embed(texts)

        raise ValueError("Unsupported EMBEDDING_PROVIDER_TYPE. Use 'nvidia' or 'http'.")

    async def _http_embed(self, texts: List[str]) -> EmbeddingResponse:
        """Call embedding service via HTTP endpoint."""
        if not settings.embedding_provider_url:
            raise RuntimeError(
                "HTTP embedding provider requires EMBEDDING_PROVIDER_URL."
            )
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.embedding_provider_url,
                json={"texts": texts},
            )
            response.raise_for_status()
            data = response.json()
            return EmbeddingResponse(
                embeddings=[
                    DenseAndSparseEmbeddings(
                        dense=item["dense"],
                        sparse=SparseVector(
                            indices=item["sparse"]["indices"],
                            values=item["sparse"]["values"],
                        ),
                    )
                    for item in data["embeddings"]
                ],
                dense_dimension=data.get("dense_dimension", self._dimension),
            )

    async def _nvidia_embed(self, texts: List[str]) -> EmbeddingResponse:
        """Call NVIDIA for dense embeddings and Pinecone inference for sparse."""
        dense_embeddings = await self._nvidia_dense_embed(texts)
        sparse_embeddings = await self._pinecone_sparse_embed(texts)

        embeddings = []
        for i, item in enumerate(dense_embeddings):
            embeddings.append(
                DenseAndSparseEmbeddings(
                    dense=item.embedding, sparse=sparse_embeddings[i]
                )
            )

        return EmbeddingResponse(
            embeddings=embeddings,
            dense_dimension=len(dense_embeddings[0].embedding),
        )

    async def _nvidia_dense_embed(self, texts: List[str]) -> List[Embedding]:
        """Call NVIDIA BGE-M3 via the OpenAI-compatible embeddings API."""
        client = self._get_nvidia_client()
        response = client.embeddings.create(
            input=texts,
            model=settings.nvidia_embedding_model,
            encoding_format="float",
            extra_body={"truncate": "NONE"},
        )

        return response.data

    async def _pinecone_sparse_embed(self, texts: List[str]) -> List[SparseVector]:
        """Get sparse embeddings from Pinecone."""
        pc = self._get_pinecone_client()

        try:
            response = pc.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=texts,
                parameters={
                    "input_type": "passage",
                    "truncate": "END",
                },
            )

            sparse_embeddings = []
            for item in response.data:
                # Pinecone sparse returns values and indices
                if item.sparse and item.sparse.values:
                    sparse_embeddings.append(
                        SparseVector(
                            indices=item.sparse.values.indices,
                            values=item.sparse.values.values,
                        )
                    )
                else:
                    # Fallback: return empty sparse if API doesn't return sparse
                    sparse_embeddings.append(SparseVector(indices=[], values=[]))

            return sparse_embeddings
        except Exception:
            # Fallback: return empty sparse on any error
            return [SparseVector(indices=[], values=[]) for _ in texts]

    def _mock_embed(self, texts: List[str]) -> EmbeddingResponse:
        """
        Generate mock embeddings for testing.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResponse with mock embeddings (dense + sparse)
        """
        import hashlib

        embeddings = []
        for text in texts:
            # Generate a deterministic hash-based dense embedding
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            dense = []
            for i in range(self._dimension):
                byte_idx = i % len(hash_bytes)
                dense.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)

            # Generate mock sparse embedding (BM25-like)
            words = text.lower().split()
            word_counts: Dict[str, int] = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Use first 10 words as sparse indices
            sparse_indices = []
            sparse_values = []
            for i, word in enumerate(list(word_counts.keys())[:10]):
                sparse_indices.append(hash(word) % 10000)
                sparse_values.append(min(word_counts[word], 1.0))

            embeddings.append(
                DenseAndSparseEmbeddings(
                    dense=dense,
                    sparse=SparseVector(indices=sparse_indices, values=sparse_values),
                )
            )

        return EmbeddingResponse(
            embeddings=embeddings,
            dense_dimension=self._dimension,
        )


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
