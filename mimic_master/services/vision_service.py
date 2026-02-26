"""Vision service for image description using GLM-4V."""

import base64
import httpx
from pathlib import Path

from mimic_master.config import settings
from mimic_master.models.vision import ImageDescription, VisionRequest


class VisionService:
    """Service for generating image descriptions using GLM-4V."""

    def __init__(self) -> None:
        self._api_key: str = settings.zhipu_api_key
        self._model: str = settings.vision_model
        self._base_url: str = settings.vision_provider_url

    async def describe_image(self, image_path: str, prompt: str | None = None) -> ImageDescription:
        """
        Describe an image using GLM-4V.

        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt

        Returns:
            ImageDescription with description and metadata
        """
        if settings.use_mock_vision:
            return self._mock_describe(image_path)

        # Encode image to base64
        image_base64 = self._encode_image(image_path)

        default_prompt = "请用中文详细描述这张图片中的D&D生物或场景，包括外观特征、颜色、姿态等关键细节。"
        actual_prompt = prompt or default_prompt

        # Build API URL - use chat/completions endpoint
        base = self._base_url.rstrip("/")
        if "chat/completions" in base:
            api_url = base
        elif "completions" in base:
            api_url = base.replace("completions", "chat/completions")
        else:
            api_url = base + "/chat/completions"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                                {"type": "text", "text": actual_prompt},
                            ],
                        }
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()

            description = data["choices"][0]["message"]["content"]

            return ImageDescription(
                image_path=image_path,
                description=description,
            )

    async def describe_images(self, image_paths: list[str]) -> list[ImageDescription]:
        """
        Describe multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of ImageDescription objects
        """
        results = []
        for path in image_paths:
            desc = await self.describe_image(path)
            results.append(desc)
        return results

    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64."""
        path = Path(image_path)
        if not path.exists():
            # Try relative to knowledge directory
            from mimic_master.config import settings
            path = settings.base_dir / "knowledge" / "raw" / image_path

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _mock_describe(self, image_path: str) -> ImageDescription:
        """
        Generate mock description for testing.

        Args:
            image_path: Path to the image file

        Returns:
            ImageDescription with mock data
        """
        # Extract filename for mock description
        filename = Path(image_path).stem
        return ImageDescription(
            image_path=image_path,
            description=f"这是{filename}的模拟描述（Mock模式）。在实际使用时，这将调用GLM-4V API获取真实描述。",
            tags=["mock", filename.lower()],
        )


# Singleton instance
_vision_service: VisionService | None = None


def get_vision_service() -> VisionService:
    """Get the singleton vision service instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
