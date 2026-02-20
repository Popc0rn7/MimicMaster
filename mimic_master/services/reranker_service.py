"""Reranker service with mock and real implementation support."""

import httpx
from typing import List, Optional

from mimic_master.config import settings
from mimic_master.models.reranker import RerankRequest, RerankResponse


class RerankerService:
    """Service for reranking documents using BGE-Reranker-v2-M3 model."""

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> RerankResponse:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_n: Number of top results to return (None for all)

        Returns:
            RerankResponse containing sorted indices and scores

        Raises:
            httpx.HTTPError: If the external service fails
        """
        if settings.use_mock_reranker:
            return self._mock_rerank(query, documents, top_n)

        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "query": query,
                "documents": documents,
            }
            if top_n is not None:
                payload["top_n"] = top_n

            response = await client.post(
                settings.reranker_provider_url,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return RerankResponse(
                results=data["results"],
                scores=data["scores"],
            )

    def _mock_rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> RerankResponse:
        """
        Mock reranking for testing.

        Uses simple keyword overlap scoring.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_n: Number of top results to return

        Returns:
            RerankResponse with mock results
        """
        query_words = set(query.lower().split())
        scores = []

        for doc in documents:
            doc_words = set(doc.lower().split())
            # Simple Jaccard-like overlap score
            if not query_words:
                scores.append(0.0)
            else:
                overlap = len(query_words & doc_words)
                scores.append(overlap / len(query_words))

        # Sort by score descending
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_n:
            indexed_scores = indexed_scores[:top_n]

        results = [i for i, _ in indexed_scores]
        scores_sorted = [score for _, score in indexed_scores]

        return RerankResponse(
            results=results,
            scores=scores_sorted,
        )


# Singleton instance
_reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    """Get the singleton reranker service instance."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
