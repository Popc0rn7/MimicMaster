"""Episodic Memory Module.

Manages retrieval of past session summaries (episodes).
Uses only dense vector retrieval (no sparse needed).
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.services.embedding_service import get_embedding_service
from mimic_master.models.memory import (
    Episode,
    RetrievalResult,
    RetrievalConfig,
)


class EpisodicRetriever:
    """
    Episodic memory retriever for past session summaries.

    Features:
    - Dense-only vector retrieval (simpler than rules retrieval)
    - Stores and retrieves session summaries
    - Tag-based filtering
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        """
        Initialize the episodic retriever.

        Args:
            config: Retrieval configuration
        """
        self._config = config or RetrievalConfig()
        self._pinecone = get_pinecone_service()
        self._embedding_service = get_embedding_service()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Episode]:
        """
        Retrieve relevant past episodes.

        Args:
            query: Query text
            top_k: Number of results to return
            namespace: Pinecone namespace (uses config default if None)
            tags: Optional tag filter
            session_id: Optional session ID filter

        Returns:
            List of retrieved episodes
        """
        top_k = top_k or self._config.top_k_episodes
        namespace = namespace or self._config.episodes_namespace

        # Step 1: Generate dense embedding for query
        embedding_response = await self._embedding_service.embed([query])
        query_embedding = embedding_response.embeddings[0]

        # Step 2: Build filter
        filter_dict: Dict[str, Any] = {}
        if tags:
            filter_dict["tags"] = {"$in": tags}
        if session_id:
            filter_dict["session_id"] = session_id

        # Step 3: Query Pinecone
        try:
            results = await self._pinecone.query(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict if filter_dict else None,
                namespace=namespace,
            )

            # Step 4: Convert to Episode objects
            episodes = []
            for result in results:
                episodes.append(
                    Episode(
                        id=result.id,
                        session_id=result.metadata.get("session_id", ""),
                        summary=result.content,
                        timestamp=datetime.fromisoformat(
                            result.metadata.get(
                                "timestamp", datetime.utcnow().isoformat()
                            )
                        ),
                        key_events=result.metadata.get("key_events", []),
                        tags=result.metadata.get("tags", []),
                    )
                )

            return episodes

        except Exception as e:
            print(f"Episodic retrieval error: {e}")
            return []

    async def add_episode(
        self,
        episode: Episode,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Add a new episode to episodic memory.

        Args:
            episode: Episode to add
            namespace: Pinecone namespace (uses config default if None)
        """
        await self.add_episodes(
            episodes=[episode],
            namespace=namespace,
        )

    async def add_episodes(
        self,
        episodes: List[Episode],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Add multiple episodes to episodic memory.

        Args:
            episodes: List of episodes to add
            namespace: Pinecone namespace
        """
        if not episodes:
            return

        namespace = namespace or self._config.episodes_namespace

        # Prepare data for upsert
        ids = [ep.id for ep in episodes]
        texts = [ep.summary for ep in episodes]
        metadata = [
            {
                "session_id": ep.session_id,
                "timestamp": ep.timestamp.isoformat(),
                "key_events": ep.key_events,
                "tags": ep.tags,
            }
            for ep in episodes
        ]

        # Upsert to Pinecone
        try:
            await self._pinecone.upsert_from_texts(
                ids=ids,
                texts=texts,
                metadata=metadata,
                namespace=namespace,
            )
            print(f"Added {len(episodes)} episodes to episodic memory")
        except Exception as e:
            print(f"Error adding episodes: {e}")

    async def get_session_episodes(
        self,
        session_id: str,
        namespace: Optional[str] = None,
    ) -> List[Episode]:
        """
        Get all episodes for a specific session.

        Args:
            session_id: Session ID
            namespace: Pinecone namespace

        Returns:
            List of episodes for the session
        """
        # Query with session_id filter
        return await self.retrieve(
            query="",
            namespace=namespace,
            session_id=session_id,
            top_k=100,  # Get all episodes for the session
        )

    async def search_by_tags(
        self,
        tags: List[str],
        query: str = "",
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> List[Episode]:
        """
        Search episodes by tags.

        Args:
            tags: Tags to filter by
            query: Optional query for relevance ranking
            top_k: Number of results to return
            namespace: Pinecone namespace

        Returns:
            List of matching episodes
        """
        return await self.retrieve(
            query=query or "session summary",
            tags=tags,
            top_k=top_k,
            namespace=namespace,
        )

    def _convert_to_episode(
        self,
        retrieval_result: RetrievalResult,
    ) -> Episode:
        """
        Convert a retrieval result to an Episode.

        Args:
            retrieval_result: Retrieval result from Pinecone

        Returns:
            Episode object
        """
        return Episode(
            id=retrieval_result.id,
            session_id=retrieval_result.metadata.get("session_id", ""),
            summary=retrieval_result.content,
            timestamp=datetime.fromisoformat(
                retrieval_result.metadata.get(
                    "timestamp", datetime.utcnow().isoformat()
                )
            ),
            key_events=retrieval_result.metadata.get("key_events", []),
            tags=retrieval_result.metadata.get("tags", []),
        )


# Singleton instance
_episodic_retriever: Optional[EpisodicRetriever] = None


def get_episodic_retriever(
    config: Optional[RetrievalConfig] = None,
) -> EpisodicRetriever:
    """Get the singleton episodic retriever instance."""
    global _episodic_retriever
    if _episodic_retriever is None or config is not None:
        _episodic_retriever = EpisodicRetriever(config)
    return _episodic_retriever
