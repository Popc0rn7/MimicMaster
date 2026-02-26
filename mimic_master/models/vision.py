"""Data models for vision service."""

from typing import Optional

from pydantic import BaseModel


class ImageDescription(BaseModel):
    """Description of an image from vision model."""

    image_path: str
    description: str
    tags: list[str] = []
    detected_text: Optional[str] = None
    confidence: float = 1.0


class VisionRequest(BaseModel):
    """Request for vision model to describe an image."""

    image_path: str
    prompt: Optional[str] = "请详细描述这张图片中的生物或场景，包括外观特征、颜色、姿态等。"


class VisionResponse(BaseModel):
    """Response from vision model."""

    descriptions: list[ImageDescription]
