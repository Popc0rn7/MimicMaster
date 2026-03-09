"""Frontend API Pydantic models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ImageType(str, Enum):
    """Image type enum for gallery."""

    MAP = "map"
    ITEM = "item"
    NPC = "npc"
    LOCATION = "location"
    OTHER = "other"


class GallerySource(str, Enum):
    """Source of gallery item."""

    UPLOAD = "upload"
    CHAT = "chat"


class CampaignStatus(str, Enum):
    """Campaign status enum."""

    NEW = "new"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# --- Campaign Models ---


class Campaign(BaseModel):
    """Campaign model."""

    id: str = Field(default="", alias="_id")
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Campaign name")
    description: str = Field(default="", description="Campaign description")
    status: CampaignStatus = Field(default=CampaignStatus.NEW)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_played_at: Optional[datetime] = None

    model_config = ConfigDict(populate_by_name=True)


class CreateCampaignRequest(BaseModel):
    """Request model for creating a campaign."""

    name: str
    description: str = ""
    file_data: Optional[str] = None  # base64 encoded JSON


# --- Session Models ---


class Session(BaseModel):
    """Session model."""

    id: str = Field(default="", alias="_id")
    campaign_id: str = Field(..., description="Campaign ID")
    user_id: str = Field(..., description="User ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True)


# --- Scene Models ---


class Scene(BaseModel):
    """Scene model."""

    id: str = Field(default="", alias="_id")
    session_id: str = Field(..., description="Session ID")
    location: str = Field(default="", description="Current location")
    time: str = Field(default="", description="Time in-game")
    weather: str = Field(default="clear", description="Weather conditions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True)


class CreateSceneRequest(BaseModel):
    """Request model for creating/updating a scene."""

    location: str
    time: str
    weather: str = "clear"


# --- Player Models ---


class SpellSlot(BaseModel):
    """Spell slot model."""

    level: int
    available: int
    max: int


class Player(BaseModel):
    """Player model."""

    id: str = Field(default="", alias="_id")
    session_id: str = Field(..., description="Session ID")
    name: str = Field(..., description="Player name")
    hp_max: int = Field(..., description="Maximum HP")
    hp_current: int = Field(..., description="Current HP")
    status_effects: list[str] = Field(default_factory=list)
    spell_slots: list[SpellSlot] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True)


class CreatePlayerRequest(BaseModel):
    """Request model for creating a player."""

    name: str
    hp_max: int
    hp_current: int
    status_effects: list[str] = Field(default_factory=list)
    spell_slots: list[SpellSlot] = Field(default_factory=list)


class UpdatePlayerRequest(BaseModel):
    """Request model for updating a player."""

    hp_max: Optional[int] = None
    hp_current: Optional[int] = None
    status_effects: Optional[list[str]] = None
    spell_slots: Optional[list[SpellSlot]] = None


# --- NPC Models ---


class NPC(BaseModel):
    """NPC model."""

    id: str = Field(default="", alias="_id")
    session_id: str = Field(..., description="Session ID")
    name: str = Field(..., description="NPC name")
    description: str = Field(default="", description="NPC description")
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True)


class CreateNPCRequest(BaseModel):
    """Request model for creating an NPC."""

    name: str
    description: str = ""
    is_active: bool = True


class UpdateNPCRequest(BaseModel):
    """Request model for updating an NPC."""

    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


# --- Gallery Models ---


class GalleryItem(BaseModel):
    """Gallery item model."""

    id: str = Field(default="", alias="_id")
    user_id: str = Field(..., description="User ID")
    url: str = Field(..., description="Image URL or base64")
    name: str = Field(..., description="Image name")
    type: ImageType = Field(default=ImageType.OTHER)
    description: Optional[str] = None
    source: GallerySource = Field(default=GallerySource.UPLOAD)
    message_id: Optional[str] = None
    pinned_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True)


class CreateGalleryItemRequest(BaseModel):
    """Request model for creating a gallery item."""

    url: str
    name: str
    type: ImageType = ImageType.OTHER
    description: Optional[str] = None
    source: GallerySource = GallerySource.UPLOAD
    message_id: Optional[str] = None


class UpdateGalleryItemRequest(BaseModel):
    """Request model for updating a gallery item."""

    name: Optional[str] = None
    type: Optional[ImageType] = None
    description: Optional[str] = None


# --- Game State Models ---


class GameState(BaseModel):
    """Full game state model."""

    session_id: str
    current_scene: Scene
    players: list[Player] = Field(default_factory=list)
    npcs: list[NPC] = Field(default_factory=list)


# --- Initialize Session Request ---


class AddPlayerRequest(BaseModel):
    """Request model for adding a player during session initialization."""

    name: str
    hp_max: int
    hp_current: int
    status_effects: list[str] = Field(default_factory=list)
    spell_slots: list[dict[str, Any]] = Field(default_factory=list)


class InitializeSessionRequest(BaseModel):
    """Request model for initializing a campaign session."""

    players: list[AddPlayerRequest]
