"""Frontend API routes for campaign, session, and gallery management."""

import base64
import json
from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query
from pymongo.errors import DuplicateKeyError

from mimic_master.database.mongodb import get_database
from mimic_master.models.frontend import (
    AddPlayerRequest,
    Campaign,
    CampaignStatus,
    CreateCampaignRequest,
    CreateGalleryItemRequest,
    CreateNPCRequest,
    CreatePlayerRequest,
    CreateSceneRequest,
    GalleryItem,
    GallerySource,
    ImageType,
    InitializeSessionRequest,
    NPC,
    Player,
    Scene,
    UpdateGalleryItemRequest,
    UpdateNPCRequest,
    UpdatePlayerRequest,
)

router = APIRouter(tags=["frontend"])

# Default user ID for development (should be replaced with actual auth)
DEFAULT_USER_ID = "default_user"


def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serializable format."""
    if doc is None:
        return None
    doc["_id"] = str(doc.get("_id", ""))
    return doc


# --- Campaign Routes ---


@router.get("/campaigns")
async def get_campaigns(
    status: Optional[CampaignStatus] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    """Get list of campaigns for the current user."""
    db = get_database()

    query = {"user_id": user_id}
    if status:
        query["status"] = status.value

    total = await db.campaigns.count_documents(query)
    campaigns = await db.campaigns.find(query).sort("updated_at", -1).skip(offset).limit(limit).to_list(limit)

    return {
        "campaigns": [serialize_doc(c) for c in campaigns],
        "total": total,
    }


@router.post("/campaigns")
async def create_campaign(
    request: CreateCampaignRequest,
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    """Create a new campaign."""
    db = get_database()

    campaign = {
        "_id": str(ObjectId()),
        "user_id": user_id,
        "name": request.name,
        "description": request.description,
        "status": CampaignStatus.NEW.value,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "last_played_at": None,
    }

    try:
        await db.campaigns.insert_one(campaign)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Campaign ID already exists")

    # Handle file_data import if provided
    if request.file_data:
        try:
            data = json.loads(base64.b64decode(request.file_data).decode("utf-8"))
            # Store file data in campaign for later processing
            await db.campaigns.update_one(
                {"_id": campaign["_id"]},
                {"$set": {"file_data": data}},
            )
        except Exception:
            pass  # Ignore invalid file_data

    return {
        "campaign": serialize_doc(campaign),
        "campaign_id": campaign["_id"],
    }


@router.get("/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str) -> dict:
    """Get campaign details."""
    db = get_database()

    campaign = await db.campaigns.find_one({"_id": campaign_id})
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Get associated sessions
    sessions = await db.sessions.find({"campaign_id": campaign_id}).to_list(10)

    # Get players if there's an active session
    players = []
    scenes = []
    dialogue_count = 0

    if sessions:
        active_session = sessions[0]  # Use most recent session
        players = await db.players.find({"session_id": active_session["_id"]}).to_list(10)
        scenes = await db.scenes.find({"session_id": active_session["_id"]}).to_list(5)
        dialogue_count = await db.dialogue_history.count_documents({"session_id": active_session["_id"]})

    return {
        "campaign": serialize_doc(campaign),
        "players": [serialize_doc(p) for p in players],
        "scenes": [serialize_doc(s) for s in scenes],
        "dialogue_count": dialogue_count,
    }


@router.post("/campaigns/{campaign_id}/initialize")
async def initialize_session(
    campaign_id: str,
    request: InitializeSessionRequest,
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    """Initialize a campaign session with players."""
    db = get_database()

    # Verify campaign exists
    campaign = await db.campaigns.find_one({"_id": campaign_id})
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Create session
    session_id = str(ObjectId())
    session = {
        "_id": session_id,
        "campaign_id": campaign_id,
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.sessions.insert_one(session)

    # Create initial scene
    scene_id = str(ObjectId())
    scene = {
        "_id": scene_id,
        "session_id": session_id,
        "location": "",
        "time": "",
        "weather": "clear",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.scenes.insert_one(scene)

    # Add players
    players = []
    for player_req in request.players:
        player = {
            "_id": str(ObjectId()),
            "session_id": session_id,
            "name": player_req.name,
            "hp_max": player_req.hp_max,
            "hp_current": player_req.hp_current,
            "status_effects": player_req.status_effects,
            "spell_slots": player_req.spell_slots,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        await db.players.insert_one(player)
        players.append(player)

    # Update campaign status
    await db.campaigns.update_one(
        {"_id": campaign_id},
        {
            "$set": {
                "status": CampaignStatus.IN_PROGRESS.value,
                "updated_at": datetime.utcnow(),
                "last_played_at": datetime.utcnow(),
            }
        },
    )

    return {
        "session_id": session_id,
        "campaign": serialize_doc(campaign),
        "players": [serialize_doc(p) for p in players],
    }


# --- Session Routes ---


@router.get("/session/{session_id}/state")
async def get_session_state(session_id: str) -> dict:
    """Get full game state for a session."""
    db = get_database()

    # Get scene
    scene = await db.scenes.find_one({"session_id": session_id})
    if not scene:
        scene = {
            "_id": "",
            "session_id": session_id,
            "location": "",
            "time": "",
            "weather": "clear",
        }

    # Get players
    players = await db.players.find({"session_id": session_id}).to_list(20)

    # Get NPCs
    npcs = await db.npcs.find({"session_id": session_id}).to_list(20)

    return {
        "session_id": session_id,
        "current_scene": serialize_doc(scene),
        "players": [serialize_doc(p) for p in players],
        "npcs": [serialize_doc(n) for n in npcs],
    }


# --- Scene Routes ---


@router.post("/session/{session_id}/scene")
async def create_scene(session_id: str, request: CreateSceneRequest) -> dict:
    """Create a new scene."""
    db = get_database()

    scene = {
        "_id": str(ObjectId()),
        "session_id": session_id,
        "location": request.location,
        "time": request.time,
        "weather": request.weather,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.scenes.insert_one(scene)

    return serialize_doc(scene)


@router.put("/session/{session_id}/scene")
async def update_scene(session_id: str, request: CreateSceneRequest) -> dict:
    """Update current scene."""
    db = get_database()

    result = await db.scenes.find_one_and_update(
        {"session_id": session_id},
        {
            "$set": {
                "location": request.location,
                "time": request.time,
                "weather": request.weather,
                "updated_at": datetime.utcnow(),
            }
        },
        sort=[("created_at", -1)],
    )

    if not result:
        raise HTTPException(status_code=404, detail="Scene not found")

    return serialize_doc(result)


# --- Player Routes ---


@router.post("/session/{session_id}/players")
async def add_player(session_id: str, request: CreatePlayerRequest) -> dict:
    """Add a player to the session."""
    db = get_database()

    player = {
        "_id": str(ObjectId()),
        "session_id": session_id,
        "name": request.name,
        "hp_max": request.hp_max,
        "hp_current": request.hp_current,
        "status_effects": request.status_effects,
        "spell_slots": request.spell_slots,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.players.insert_one(player)

    return serialize_doc(player)


@router.put("/session/{session_id}/players/{player_name}")
async def update_player(
    session_id: str,
    player_name: str,
    request: UpdatePlayerRequest,
) -> dict:
    """Update a player's state."""
    db = get_database()

    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()

    result = await db.players.find_one_and_update(
        {"session_id": session_id, "name": player_name},
        {"$set": update_data},
        return_document=True,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Player not found")

    return serialize_doc(result)


@router.delete("/session/{session_id}/players/{player_name}")
async def delete_player(session_id: str, player_name: str) -> dict:
    """Remove a player from the session."""
    db = get_database()

    result = await db.players.delete_one({"session_id": session_id, "name": player_name})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Player not found")

    return {"message": "Player removed"}


# --- NPC Routes ---


@router.post("/session/{session_id}/npcs")
async def add_npc(session_id: str, request: CreateNPCRequest) -> dict:
    """Add an NPC to the session."""
    db = get_database()

    npc = {
        "_id": str(ObjectId()),
        "session_id": session_id,
        "name": request.name,
        "description": request.description,
        "is_active": request.is_active,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.npcs.insert_one(npc)

    return serialize_doc(npc)


@router.put("/session/{session_id}/npcs/{npc_name}")
async def update_npc(
    session_id: str,
    npc_name: str,
    request: UpdateNPCRequest,
) -> dict:
    """Update an NPC."""
    db = get_database()

    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()

    result = await db.npcs.find_one_and_update(
        {"session_id": session_id, "name": npc_name},
        {"$set": update_data},
        return_document=True,
    )

    if not result:
        raise HTTPException(status_code=404, detail="NPC not found")

    return serialize_doc(result)


@router.delete("/session/{session_id}/npcs/{npc_name}")
async def delete_npc(session_id: str, npc_name: str) -> dict:
    """Remove an NPC from the session."""
    db = get_database()

    result = await db.npcs.delete_one({"session_id": session_id, "name": npc_name})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="NPC not found")

    return {"message": "NPC removed"}


# --- Gallery Routes ---


@router.get("/gallery")
async def get_gallery(
    type: Optional[ImageType] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    """Get gallery items for the current user."""
    db = get_database()

    query = {"user_id": user_id}
    if type:
        query["type"] = type.value

    total = await db.gallery.count_documents(query)
    items = await db.gallery.find(query).sort("pinned_at", -1).skip(offset).limit(limit).to_list(limit)

    return {
        "items": [serialize_doc(i) for i in items],
        "total": total,
    }


@router.post("/gallery")
async def create_gallery_item(
    request: CreateGalleryItemRequest,
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    """Add an item to the gallery."""
    db = get_database()

    item = {
        "_id": str(ObjectId()),
        "user_id": user_id,
        "url": request.url,
        "name": request.name,
        "type": request.type.value if request.type else ImageType.OTHER.value,
        "description": request.description,
        "source": request.source.value if request.source else GallerySource.UPLOAD.value,
        "message_id": request.message_id,
        "pinned_at": datetime.utcnow(),
        "created_at": datetime.utcnow(),
    }
    await db.gallery.insert_one(item)

    return serialize_doc(item)


@router.put("/gallery/{item_id}")
async def update_gallery_item(
    item_id: str,
    request: UpdateGalleryItemRequest,
) -> dict:
    """Update a gallery item."""
    db = get_database()

    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    if "type" in update_data:
        update_data["type"] = update_data["type"].value

    result = await db.gallery.find_one_and_update(
        {"_id": item_id},
        {"$set": update_data},
        return_document=True,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Gallery item not found")

    return serialize_doc(result)


@router.delete("/gallery/{item_id}")
async def delete_gallery_item(item_id: str) -> dict:
    """Delete a gallery item."""
    db = get_database()

    result = await db.gallery.delete_one({"_id": item_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Gallery item not found")

    return {"message": "Gallery item deleted"}


@router.post("/gallery/upload")
async def upload_gallery_item(
    file_data: str = Query(..., description="Base64 encoded image data"),
    name: str = Query(..., description="Image name"),
    type: ImageType = Query(ImageType.OTHER),
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    """Upload an image to the gallery."""
    db = get_database()

    item = {
        "_id": str(ObjectId()),
        "user_id": user_id,
        "url": file_data,  # Store base64 directly (for now)
        "name": name,
        "type": type.value,
        "description": None,
        "source": GallerySource.UPLOAD.value,
        "message_id": None,
        "pinned_at": datetime.utcnow(),
        "created_at": datetime.utcnow(),
    }
    await db.gallery.insert_one(item)

    return {
        "id": item["_id"],
        "url": item["url"],
        "name": item["name"],
    }
