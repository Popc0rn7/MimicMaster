"""MongoDB database connection and client management."""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from mimic_master.config import settings

# Global database client
_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


async def init_mongodb() -> None:
    """Initialize MongoDB connection."""
    global _client, _database

    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _database = _client[settings.mongodb_database]

    # Verify connection
    await _client.admin.command("ping")

    # Create indexes
    await _create_indexes()


async def _create_indexes() -> None:
    """Create MongoDB indexes for better query performance."""
    if _database is None:
        return

    # Campaigns indexes
    await _database.campaigns.create_index("user_id")
    await _database.campaigns.create_index([("updated_at", -1)])

    # Sessions indexes
    await _database.sessions.create_index("campaign_id")
    await _database.sessions.create_index("user_id")

    # Players indexes
    await _database.players.create_index([("session_id", 1), ("name", 1)])

    # NPCs indexes
    await _database.npcs.create_index([("session_id", 1), ("name", 1)])

    # Scenes indexes
    await _database.scenes.create_index("session_id")

    # Dialogue history indexes
    await _database.dialogue_history.create_index(
        [("session_id", 1), ("timestamp", -1)]
    )

    # Gallery indexes
    await _database.gallery.create_index("user_id")
    await _database.gallery.create_index([("user_id", 1), ("pinned_at", -1)])


async def close_mongodb() -> None:
    """Close MongoDB connection."""
    global _client, _database

    if _client:
        _client.close()
        _client = None
        _database = None


def get_database() -> AsyncIOMotorDatabase:
    """Get the MongoDB database instance."""
    if _database is None:
        raise RuntimeError("MongoDB not initialized. Call init_mongodb() first.")
    return _database
