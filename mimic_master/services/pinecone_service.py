"""Pinecone vector database service with singleton pattern."""

from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from mimic_master.config import settings
from mimic_master.models.embeddings import EmbeddingResponse
from mimic_master.models.retrieval import RetrievedDocument

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
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"Index '{self._index_name}' created successfully.")

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadata: Optional[List[dict]] = None,
        namespace: str = "",
    ) -> None:
        """
        Upsert documents into the index.

        Args:
            ids: List of document IDs
            embeddings: List of embedding vectors
            contents: List of document contents
            metadata: Optional list of metadata dictionaries
            namespace: Namespace for the documents
        """
        if not (len(ids) == len(embeddings) == len(contents)):
            raise ValueError("ids, embeddings, and contents must have the same length")

        if metadata is None:
            metadata = [{}] * len(ids)

        vectors = []
        for i, (id_, embedding) in enumerate(zip(ids, embeddings)):
            vector_metadata = {
                "content": contents[i],
                **metadata[i],
            }
            vectors.append({
                "id": id_,
                "values": embedding,
                "metadata": vector_metadata,
            })

        index = self.client.Index(self._index_name)
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Upserted {len(vectors)} vectors to index '{self._index_name}'.")

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[dict] = None,
        namespace: str = "",
    ) -> List[RetrievedDocument]:
        """
        Query the index for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            namespace: Namespace to query

        Returns:
            List of retrieved documents with scores
        """
        if not settings.is_pinecone_configured:
            raise RuntimeError("Pinecone is not configured.")

        index = self.client.Index(self._index_name)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            namespace=namespace,
            include_metadata=True,
        )

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
        from mimic_master.models.embeddings import EmbeddingRequest

        response = await embedding_service.embed(texts)
        await self.upsert(
            ids=ids,
            embeddings=response.embeddings,
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
