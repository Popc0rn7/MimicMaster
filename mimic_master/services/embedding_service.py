"""Embedding service with mock and real implementation support."""

import httpx
from typing import List, Dict

from mimic_master.config import settings
from mimic_master.models.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    DenseAndSparseEmbeddings,
    SparseVector,
)


class EmbeddingService:
    """Service for generating text embeddings using BGE-M3 model."""

    def __init__(self) -> None:
        self._dimension: int = settings.embedding_dimension

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
        if settings.use_mock_embedding:
            return self._mock_embed(texts)

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
