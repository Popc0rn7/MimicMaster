"""
Setup script for Pinecone index.

This script creates the Pinecone index with the correct configuration
for hybrid search (dense + sparse vectors) using dotproduct metric.
"""

import asyncio
from mimic_master.services.pinecone_service import get_pinecone_service
from mimic_master.config import settings


async def main() -> None:
    """Create Pinecone index."""
    print(f"Creating Pinecone index: {settings.pinecone_index}")
    print(f"Dimension: {settings.embedding_dimension}")
    print(f"Metric: dotproduct (for hybrid search)")

    pinecone_service = get_pinecone_service()

    # Check if index exists
    existing_indexes = pinecone_service.client.list_indexes().names()
    if settings.pinecone_index in existing_indexes:
        print(f"\nIndex '{settings.pinecone_index}' already exists.")
        print(f"Skipping creation.")
        return

    # Create index
    print("\nCreating index...")
    await pinecone_service.create_index(
        cloud="aws",
        region="us-east-1",
    )

    print("\nIndex created successfully!")
    print(f"Name: {settings.pinecone_index}")
    print(f"Dimension: {settings.embedding_dimension}")
    print(f"Metric: dotproduct")


if __name__ == "__main__":
    asyncio.run(main())
