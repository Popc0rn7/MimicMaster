"""Static knowledge retrieval through configured embedding and Pinecone services."""

from typing import Dict, List, Optional

from mimic_master.models.memory import RetrievalConfig, RetrievalResult
from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.services.reranker_service import get_reranker_service


class HybridKnowledgeRetriever:
    """Knowledge retriever backed by the configured embedding provider and Pinecone."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._config = config or RetrievalConfig()
        self._embedding = get_embedding_service()
        self._pinecone = get_pinecone_service()
        self._reranker = get_reranker_service()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve relevant knowledge using dense+sparse Pinecone search."""
        top_k = top_k or self._config.top_k_rules
        namespace = namespace or self._config.rules_namespace

        embedding_response = await self._embedding.embed([query])
        query_embedding = embedding_response.embeddings[0]
        sparse_vector = {
            "indices": query_embedding.sparse.indices,
            "values": query_embedding.sparse.values,
        }

        documents = await self._pinecone.query(
            query_embedding=query_embedding.dense,
            sparse_vector=sparse_vector,
            top_k=top_k * 2 if use_rerank else top_k,
            namespace=namespace,
        )

        results = [
            RetrievalResult(
                id=doc.id,
                content=doc.content,
                score=doc.score,
                metadata=doc.metadata,
            )
            for doc in documents
        ]

        if use_rerank and len(results) > 1:
            return await self._rerank(query, results, top_k)
        return results[:top_k]

    async def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Rerank retrieved results with the configured reranker service."""
        rerank_response = await self._reranker.rerank(
            query=query,
            documents=[result.content for result in results],
            top_n=top_k,
        )

        reranked = []
        for rank, idx in enumerate(rerank_response.results):
            if idx >= len(results):
                continue
            result = results[idx]
            score = rerank_response.scores[rank]
            reranked.append(
                RetrievalResult(
                    id=result.id,
                    content=result.content,
                    score=score,
                    metadata=result.metadata,
                )
            )
        return reranked

    async def index_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Index knowledge documents using the configured embedding provider."""
        await self._pinecone.upsert_from_texts(
            ids=ids,
            texts=texts,
            metadata=metadata,
            namespace=namespace or self._config.rules_namespace,
        )


_hybrid_knowledge_retriever: Optional[HybridKnowledgeRetriever] = None


def get_hybrid_knowledge_retriever(
    config: Optional[RetrievalConfig] = None,
) -> HybridKnowledgeRetriever:
    """Get the singleton hybrid knowledge retriever instance."""
    global _hybrid_knowledge_retriever
    if _hybrid_knowledge_retriever is None or config is not None:
        _hybrid_knowledge_retriever = HybridKnowledgeRetriever(config)
    return _hybrid_knowledge_retriever
