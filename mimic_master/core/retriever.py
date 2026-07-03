"""D&D specific retriever implementation."""

from typing import List, Optional, Dict, Any

from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.services.reranker_service import get_reranker_service

from mimic_master.models.retrieval import RetrievedDocument


class DandDRetriever:
    """
    D&D specific document retriever.

    Handles retrieval of D&D rules, lore, and reference materials
    with support for different namespaces (campaigns, worlds, etc.).
    """

    def __init__(
        self,
        default_namespace: str = "",
        rerank_threshold: int = 3,
    ) -> None:
        """
        Initialize the D&D retriever.

        Args:
            default_namespace: Default namespace for queries
            rerank_threshold: Minimum number of results to trigger reranking
        """
        self.default_namespace = default_namespace
        self.rerank_threshold = rerank_threshold
        self._pinecone = get_pinecone_service()
        self._embedding = get_embedding_service()
        self._reranker = get_reranker_service()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_rerank: bool = True,
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a D&D query.

        Args:
            query: The search query
            top_k: Number of results to retrieve
            namespace: Namespace to search (uses default if None)
            filters: Metadata filters for the search
            use_rerank: Whether to use reranking for results

        Returns:
            List of retrieved documents
        """
        namespace = namespace or self.default_namespace

        # Generate query embedding
        embedding_response = await self._embedding.embed([query])
        query_embedding = embedding_response.embeddings[0]
        sparse_vector = {
            "indices": query_embedding.sparse.indices,
            "values": query_embedding.sparse.values,
        }

        # Query Pinecone
        results = await self._pinecone.query(
            query_embedding=query_embedding.dense,
            sparse_vector=sparse_vector,
            top_k=top_k,
            filter_dict=filters,
            namespace=namespace,
        )

        # Rerank if configured
        if use_rerank and len(results) >= self.rerank_threshold:
            results = await self._rerank_results(query, results)

        return results

    async def _rerank_results(
        self,
        query: str,
        results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        Rerank retrieval results.

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Reranked results
        """
        doc_texts = [doc.content for doc in results]
        rerank_response = await self._reranker.rerank(
            query=query,
            documents=doc_texts,
            top_n=len(results),
        )

        # Reorder results based on reranking
        reranked = [results[idx] for idx in rerank_response.results]
        return reranked

    async def index_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Index D&D documents into the vector database.

        Args:
            ids: Document IDs
            texts: Document contents
            metadata: Optional metadata for each document
            namespace: Namespace for indexing
        """
        namespace = namespace or self.default_namespace
        await self._pinecone.upsert_from_texts(
            ids=ids,
            texts=texts,
            metadata=metadata,
            namespace=namespace,
        )


# Preset retrievers for different D&D content types
class DandDRetrieverFactory:
    """Factory for creating pre-configured D&D retrievers."""

    @staticmethod
    def create_rules_retriever(namespace: str = "rules") -> DandDRetriever:
        """Create a retriever for D&D rules."""
        return DandDRetriever(
            default_namespace=namespace,
            rerank_threshold=5,
        )

    @staticmethod
    def create_lore_retriever(namespace: str = "lore") -> DandDRetriever:
        """Create a retriever for D&D lore."""
        return DandDRetriever(
            default_namespace=namespace,
            rerank_threshold=3,
        )

    @staticmethod
    def create_monster_retriever(namespace: str = "monsters") -> DandDRetriever:
        """Create a retriever for D&D monsters."""
        return DandDRetriever(
            default_namespace=namespace,
            rerank_threshold=5,
        )

    @staticmethod
    def create_spell_retriever(namespace: str = "spells") -> DandDRetriever:
        """Create a retriever for D&D spells."""
        return DandDRetriever(
            default_namespace=namespace,
            rerank_threshold=5,
        )
