"""Pinecone vector database service with singleton pattern."""

from typing import List, Optional, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from mimic_master.config import settings
from mimic_master.models.embeddings import EmbeddingResponse
from mimic_master.models.retrieval import RetrievedDocument
from mimic_master.models.embeddings import DenseAndSparseEmbeddings, SparseVector

from mimic_master.services.embedding_service import get_embedding_service


class PineconeService:
    """Service for interacting with Pinecone vector database."""

    def __init__(self) -> None:
        """Initialize Pinecone client (singleton pattern)."""
        self._client: Optional[Pinecone] = None
        self._index_name: str = settings.pinecone_index
        self._dimension: int = settings.embedding_dimension

    @property
    def client(self) -> Pinecone:
        """Get or create Pinecone client (lazy initialization)."""
        if self._client is None:
            if not settings.is_pinecone_configured:
                raise RuntimeError(
                    "Pinecone is not configured. Please set PINECONE_API_KEY and PINECONE_INDEX."
                )
            self._client = Pinecone(api_key=settings.pinecone_api_key)
        return self._client

    async def create_index(
        self,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        """
        Create a new Pinecone index.

        Args:
            cloud: Cloud provider (default: aws)
            region: Region for the index (default: us-east-1)
        """
        if self._index_name in self.client.list_indexes().names():
            print(f"Index '{self._index_name}' already exists.")
            return

        self.client.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric="dotproduct",  # Use dotproduct for hybrid search
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"Index '{self._index_name}' created successfully.")

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[DenseAndSparseEmbeddings],
        contents: List[str],
        metadata: Optional[List[dict]] = None,
        namespace: str = "",
    ) -> None:
        """
        Upsert documents into index with dense + sparse embeddings.

        Args:
            ids: List of document IDs
            embeddings: List of dense and sparse embeddings
            contents: List of document contents
            metadata: Optional list of metadata dictionaries
            namespace: Namespace for the documents
        """
        if not (len(ids) == len(embeddings) == len(contents)):
            raise ValueError("ids, embeddings, and contents must have the same length")

        if metadata is None:
            metadata = [{}] * len(ids)

        vectors = []
        for i, (id_, emb) in enumerate(zip(ids, embeddings)):
            vector_metadata = {
                "content": contents[i],
                **metadata[i],
            }

            # Extract dense vector - handle both Pydantic model and dict
            if hasattr(emb, 'dense'):
                dense_vals = list(emb.dense)
            elif isinstance(emb, dict) and 'dense' in emb:
                dense_vals = emb['dense']
            else:
                dense_vals = emb['dense'] if hasattr(emb, 'dense') else emb['dense']

            # Extract sparse vector - handle both Pydantic model and dict
            if hasattr(emb, 'sparse') and hasattr(emb.sparse, 'indices'):
                sparse_indices = list(emb.sparse.indices)
                sparse_values = list(emb.sparse.values)
            elif isinstance(emb, dict) and 'sparse' in emb:
                sparse = emb['sparse']
                sparse_indices = sparse['indices'] if isinstance(sparse, dict) else sparse.indices
                sparse_values = sparse['values'] if isinstance(sparse, dict) else sparse.values
            else:
                # Fallback
                sparse = emb.get('sparse', {})
                sparse_indices = sparse.get('indices', [])
                sparse_values = sparse.get('values', [])

            # Build vector with both dense and sparse components
            vector_data: Dict[str, Any] = {
                "id": id_,
                "values": dense_vals,
                "sparse_values": {
                    "indices": sparse_indices,
                    "values": sparse_values,
                },
                "metadata": vector_metadata,
            }
            vectors.append(vector_data)

        index = self.client.Index(self._index_name)
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Upserted {len(vectors)} vectors to index '{self._index_name}'.")

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        sparse_vector: Optional[Dict[str, Any]] = None,
        filter_dict: Optional[dict] = None,
        namespace: str = "",
    ) -> List[RetrievedDocument]:
        """
        Query index with hybrid search (dense + sparse).

        Args:
            query_embedding: Dense query embedding vector
            top_k: Number of results to return
            sparse_vector: Optional sparse vector for hybrid search
            filter_dict: Optional metadata filter
            namespace: Namespace to query

        Returns:
            List of retrieved documents with scores
        """
        if not settings.is_pinecone_configured:
            raise RuntimeError("Pinecone is not configured.")

        index = self.client.Index(self._index_name)

        # Build query parameters
        query_params: Dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "filter": filter_dict,
            "namespace": namespace,
            "include_metadata": True,
        }

        # Add sparse vector if provided (hybrid search)
        if sparse_vector is not None:
            query_params["sparse_vector"] = sparse_vector

        results = index.query(**query_params)

        documents = []
        for match in results.matches:
            if match.metadata:
                documents.append(
                    RetrievedDocument(
                        id=match.id,
                        content=match.metadata.get("content", ""),
                        score=match.score or 0.0,
                        metadata={k: v for k, v in match.metadata.items() if k != "content"},
                    )
                )

        return documents

    async def upsert_from_texts(
        self,
        ids: List[str],
        texts: List[str],
        metadata: Optional[List[dict]] = None,
        namespace: str = "",
    ) -> None:
        """
        Upsert documents from text (embeddings generated internally).

        Args:
            ids: List of document IDs
            texts: List of document contents
            metadata: Optional list of metadata dictionaries
            namespace: Namespace for the documents
        """
        embedding_service = get_embedding_service()

        response = await embedding_service.embed(texts)

        # Convert to plain dicts for Pinecone compatibility
        embeddings = []
        for emb in response.embeddings:
            if hasattr(emb, 'model_dump'):
                # It's a Pydantic model
                embeddings.append(emb.model_dump())
            else:
                # It's already a dict
                embeddings.append(emb)

        await self.upsert(
            ids=ids,
            embeddings=embeddings,
            contents=texts,
            metadata=metadata,
            namespace=namespace,
        )


# Singleton instance
_pinecone_service: PineconeService | None = None


def get_pinecone_service() -> PineconeService:
    """Get the singleton Pinecone service instance."""
    global _pinecone_service
    if _pinecone_service is None:
        _pinecone_service = PineconeService()
    return _pinecone_service
