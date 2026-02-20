"""Embedding service with mock and real implementation support."""

import httpx
from typing import List

from mimic_master.config import settings
from mimic_master.models.embeddings import EmbeddingRequest, EmbeddingResponse


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
            EmbeddingResponse containing the embedding vectors

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
                embeddings=data["embeddings"],
                dimension=data.get("dimension", self._dimension),
            )

    def _mock_embed(self, texts: List[str]) -> EmbeddingResponse:
        """
        Generate mock embeddings for testing.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResponse with mock embeddings
        """
        import hashlib

        embeddings = []
        for text in texts:
            # Generate a deterministic hash-based embedding
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Expand to the required dimension
            embedding = []
            for i in range(self._dimension):
                byte_idx = i % len(hash_bytes)
                embedding.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)
            embeddings.append(embedding)

        return EmbeddingResponse(
            embeddings=embeddings,
            dimension=self._dimension,
        )


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
