"""Static Knowledge Retrieval Module.

Implements hybrid search (Dense + Sparse) with local BGE-M3 model
and local reranking with BGE-Reranker-v2-M3.
"""

from typing import List, Optional, Dict, Any, Tuple

try:
    import numpy as np
except ImportError:
    np = None

from mimic_master.config import settings
from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.models.memory import RetrievalResult, RetrievalConfig

# Singleton model instances (lazy loading)
_embedding_model: Any = None
_reranker_model: Any = None


def _load_embedding_model() -> Any:
    """
    Load the BGE-M3 embedding model (singleton).

    Returns:
        The FlagEmbedding model instance
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            _embedding_model = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=True,
                device="cpu",  # Change to "cuda" if GPU available
            )
            print("BGE-M3 embedding model loaded (Dense + Sparse)")
        except ImportError:
            print("FlagEmbedding not available, using mock embeddings")
            raise ImportError(
                "FlagEmbedding is required for local BGE-M3 model. "
                "Install with: uv add FlagEmbedding"
            )
    return _embedding_model


def _load_reranker_model() -> Any:
    """
    Load the BGE-Reranker-v2-M3 model (singleton).

    Returns:
        The FlagReranker model instance
    """
    global _reranker_model
    if _reranker_model is None:
        try:
            from FlagEmbedding import FlagReranker
            _reranker_model = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=True,
                device="cpu",  # Change to "cuda" if GPU available
            )
            print("BGE-Reranker-v2-M3 model loaded")
        except ImportError:
            print("FlagEmbedding not available for reranking")
            raise ImportError(
                "FlagEmbedding is required for local reranker model. "
                "Install with: uv add FlagEmbedding"
            )
    return _reranker_model


class HybridKnowledgeRetriever:
    """
    Static knowledge retriever with hybrid search and reranking.

    Features:
    - Dense + Sparse vector generation with BGE-M3
    - Hybrid search on Pinecone
    - Local reranking with BGE-Reranker-v2-M3
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        """
        Initialize the hybrid knowledge retriever.

        Args:
            config: Retrieval configuration
        """
        self._config = config or RetrievalConfig()
        self._pinecone = get_pinecone_service()
        self._embedding_model: Optional[Any] = None
        self._reranker_model: Optional[Any] = None

    @property
    def embedding_model(self) -> Any:
        """Get or lazy-load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = _load_embedding_model()
        return self._embedding_model

    @property
    def reranker_model(self) -> Any:
        """Get or lazy-load the reranker model."""
        if self._reranker_model is None:
            self._reranker_model = _load_reranker_model()
        return self._reranker_model

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant rules/knowledge using hybrid search.

        Args:
            query: Query text
            top_k: Number of results to return
            namespace: Pinecone namespace (uses config default if None)
            use_rerank: Whether to use local reranking

        Returns:
            List of retrieved results with scores
        """
        top_k = top_k or self._config.top_k_rules
        namespace = namespace or self._config.rules_namespace

        # Step 1: Generate Dense + Sparse embeddings
        try:
            dense_vec, sparse_vec = self._encode_query(query)
        except ImportError:
            # Fall back to mock if models not available
            return await self._mock_retrieve(query, top_k, namespace)

        # Step 2: Hybrid search on Pinecone
        initial_results = await self._hybrid_search(
            dense_vec,
            sparse_vec,
            top_k * 2,  # Retrieve more for reranking
            namespace,
        )

        if not initial_results:
            return []

        # Step 3: Rerank with local model
        if use_rerank and len(initial_results) > 1:
            results = await self._rerank(query, initial_results, top_k)
        else:
            results = initial_results[:top_k]

        return results

    def _encode_query(self, query: str) -> Tuple[List[float], Dict[int, float]]:
        """
        Encode query with BGE-M3 (Dense + Sparse).

        Args:
            query: Query text

        Returns:
            Tuple of (dense_vector, sparse_vector)
        """
        # Encode with both dense and sparse
        output = self.embedding_model.encode(
            query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vec = output["dense_vecs"][0].tolist()
        sparse_vec = output["lexical_weights"][0]

        return dense_vec, sparse_vec

    async def _hybrid_search(
        self,
        dense_vec: List[float],
        sparse_vec: Dict[int, float],
        top_k: int,
        namespace: str,
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search on Pinecone.

        Args:
            dense_vec: Dense vector
            sparse_vec: Sparse vector (dict of token_id: weight)
            top_k: Number of results
            namespace: Pinecone namespace

        Returns:
            List of retrieval results
        """
        try:
            from pinecone import VectorType

            # Convert sparse vector to Pinecone format
            # Pinecone expects {"indices": [...], "values": [...]}
            sparse_pinecone: VectorType = {
                "indices": list(sparse_vec.keys()),
                "values": list(sparse_vec.values()),
            }

            results = await self._pinecone.query(
                query_embedding=dense_vec,
                sparse_vector=sparse_pinecone,
                top_k=top_k,
                namespace=namespace,
            )

            return [
                RetrievalResult(
                    id=doc.id,
                    content=doc.content,
                    score=doc.score,
                    metadata=doc.metadata,
                )
                for doc in results
            ]
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []

    async def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Rerank results with local BGE-Reranker model.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        try:
            # Prepare pairs for reranking
            pairs = [[query, result.content] for result in results]

            # Compute scores
            scores = self.reranker_model.compute_score(pairs, normalize=True)

            # If compute_score returns a single float, it computed all pairs
            if isinstance(scores, float):
                scores = [scores]  # This shouldn't happen with multiple pairs

            # Sort by score descending
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top_k results with updated scores
            reranked = []
            for idx, score in indexed_scores[:top_k]:
                result = results[idx]
                reranked.append(
                    RetrievalResult(
                        id=result.id,
                        content=result.content,
                        score=float(score),
                        metadata=result.metadata,
                    )
                )

            return reranked

        except Exception as e:
            print(f"Reranking error: {e}")
            return results[:top_k]

    async def _mock_retrieve(
        self,
        query: str,
        top_k: int,
        namespace: str,
    ) -> List[RetrievalResult]:
        """
        Mock retrieval for testing without models.

        Args:
            query: Query text
            top_k: Number of results
            namespace: Namespace

        Returns:
            Mock retrieval results
        """
        from mimic_master.services.embedding_service import get_embedding_service

        # Use mock embedding service
        embedding_service = get_embedding_service()
        response = await embedding_service.embed([query])

        # Extract dense vector from response
        emb = response.embeddings[0]
        dense_vector = emb.dense if hasattr(emb, 'dense') else emb['dense']

        # Query Pinecone with dense vector
        results = await self._pinecone.query(
            query_embedding=dense_vector,
            top_k=top_k,
            namespace=namespace,
        )

        return [
            RetrievalResult(
                id=doc.id,
                content=doc.content,
                score=doc.score,
                metadata=doc.metadata,
            )
            for doc in results
        ]

    async def index_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Index knowledge documents with Dense + Sparse embeddings.

        Args:
            ids: Document IDs
            texts: Document contents
            metadata: Optional metadata
            namespace: Pinecone namespace
        """
        try:
            # Encode all texts
            outputs = self.embedding_model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                batch_size=32,
            )

            dense_embeddings = outputs["dense_vecs"]
            sparse_embeddings = outputs["lexical_weights"]

            # Upsert to Pinecone
            if metadata is None:
                metadata = [{}] * len(ids)

            vectors = []
            for i, (id_, dense, sparse) in enumerate(zip(ids, dense_embeddings, sparse_embeddings)):
                vectors.append({
                    "id": id_,
                    "values": dense.tolist(),
                    "sparse_values": sparse,
                    "metadata": {
                        "content": texts[i],
                        **metadata[i],
                    },
                })

            await self._pinecone.upsert(
                ids=ids,
                embeddings=[v["values"] for v in vectors],
                contents=texts,
                metadata=metadata,
                namespace=namespace or self._config.rules_namespace,
            )

        except ImportError:
            print("BGE-M3 model not available, using mock embedding")
            await self._pinecone.upsert_from_texts(
                ids=ids,
                texts=texts,
                metadata=metadata,
                namespace=namespace or self._config.rules_namespace,
            )


# Singleton instance
_hybrid_knowledge_retriever: Optional[HybridKnowledgeRetriever] = None


def get_hybrid_knowledge_retriever(
    config: Optional[RetrievalConfig] = None,
) -> HybridKnowledgeRetriever:
    """Get the singleton hybrid knowledge retriever instance."""
    global _hybrid_knowledge_retriever
    if _hybrid_knowledge_retriever is None or config is not None:
        _hybrid_knowledge_retriever = HybridKnowledgeRetriever(config)
    return _hybrid_knowledge_retriever
