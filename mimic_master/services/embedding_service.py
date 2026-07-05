"""Embedding service with OpenAI-compatible provider support."""

from __future__ import annotations

import importlib.util
import os
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


def normalize_proxy_environment(socksio_available: bool | None = None) -> list[str]:
    """Remove SOCKS all-proxy settings when httpx lacks SOCKS support."""

    if socksio_available is None:
        socksio_available = importlib.util.find_spec("socksio") is not None
    if socksio_available:
        return []

    removed = []
    for key in ("all_proxy", "ALL_PROXY"):
        value = os.environ.get(key, "")
        if value.lower().startswith(("socks4://", "socks5://")):
            os.environ.pop(key, None)
            removed.append(key)
    return removed


class EmbeddingService:
    """Service for generating text embeddings using BGE-M3 model."""

    def __init__(self) -> None:
        self._dimension: int = settings.embedding_dimension
        self._embedding_client: Optional["OpenAI"] = None
        self._pinecone_client: Optional["Pinecone"] = None

    def _get_embedding_client(self) -> "OpenAI":
        """Get or create the selected OpenAI-compatible embedding client."""
        if not settings.is_embedding_configured:
            raise RuntimeError(
                f"{settings.embedding_backend} embedding requires api key, "
                "base URL, and model settings."
            )
        if self._embedding_client is None:
            normalize_proxy_environment()
            self._embedding_client = OpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )
        return self._embedding_client

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
            openai.OpenAIError: If the external service fails
        """
        return await self._openai_compatible_embed(texts)

    async def _openai_compatible_embed(self, texts: List[str]) -> EmbeddingResponse:
        """Call the selected OpenAI-compatible backend and add sparse embeddings."""
        dense_embeddings = await self._dense_embed(texts)
        if not dense_embeddings:
            raise RuntimeError(
                f"{settings.embedding_backend} returned no dense embeddings "
                f"for {len(texts)} texts."
            )
        if len(dense_embeddings) != len(texts):
            raise RuntimeError(
                f"{settings.embedding_backend} returned {len(dense_embeddings)} "
                f"dense embeddings for {len(texts)} texts."
            )
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

    async def _dense_embed(self, texts: List[str]) -> List[Embedding]:
        """Call the selected backend through the OpenAI-compatible embeddings API."""
        client = self._get_embedding_client()
        response = client.embeddings.create(
            input=texts,
            model=settings.embedding_model,
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
