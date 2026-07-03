"""Image processing script.

Only does image cleaning for knowledge JSONL:
- Detects one or more `[IMG:...]` tags in each entry's `text`
- Calls the vision model to produce description(s)
- Removes the IMG tag(s) and prepends description line(s)
- Writes the result into `knowledge/use/<same filename as raw>`

Usage:
    # process all: auto-discover knowledge/raw/rag_*.jsonl
    python scripts/process_images.py

    # process specific file(s): source key, stem, or filename
    python scripts/process_images.py --source mm
    python scripts/process_images.py --source rag_mm
    python scripts/process_images.py --source rag_mm.jsonl
    python scripts/process_images.py --source mm --source dmg

    # quick testing: limit entries per file
    python scripts/process_images.py --limit 10
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mimic_master.services.vision_service import get_vision_service

# Base paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "knowledge" / "raw"
ENHANCE_DIR = BASE_DIR / "knowledge" / "enhanced"
USE_DIR = BASE_DIR / "knowledge" / "use"
IMG_DIR = RAW_DIR / "img"


class ImageProcessor:
    """Processor for cleaning `[IMG:...]` tags with vision descriptions."""

    _IMG_PATTERN = re.compile(r"\[IMG:(.*?)\]")

    def __init__(self, output_dir: Path | None = None):
        self.vision_service = get_vision_service()
        self.output_dir = output_dir or USE_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_image_paths(self, text: str) -> List[str]:
        """Extract all image paths from `[IMG:...]` tags (order preserved)."""
        return [
            m.group(1).strip()
            for m in self._IMG_PATTERN.finditer(text)
            if m.group(1).strip()
        ]

    def strip_img_tags(self, text: str) -> str:
        """Remove all `[IMG:...]` tags from text."""
        return self._IMG_PATTERN.sub("", text).strip()

    async def process_image(self, image_path: str) -> str:
        """Process a single image and return description."""
        if not self.vision_service.image_exists(image_path):
            # Try to infer description from filename
            print(f"  Warning: Image file not found, inferring from name: {image_path}")
            try:
                result = await self.vision_service.describe_by_name(image_path)
                return f"[图片描述: {result.description}]"
            except Exception as e:
                print(f"  Error inferring description: {e}")
                return f"[图片: {image_path}]"

        try:
            result = await self.vision_service.describe_image(image_path)
            return f"[图片描述: {result.description}]"
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            return f"[图片: {image_path}]"

    async def process_text(self, text: str) -> str:
        """Clean one entry text: strip IMG tags and prepend vision description(s)."""
        image_paths = self.extract_image_paths(text)
        if not image_paths:
            return text

        cleaned_text = self.strip_img_tags(text)
        description_lines: List[str] = []
        for image_path in image_paths:
            description_lines.append(await self.process_image(image_path))

        return "\n".join(description_lines) + "\n" + cleaned_text

    async def process_jsonl(self, source_name: str, limit: Optional[int] = None) -> int:
        """Process a single JSONL file (raw -> use) with image cleaning."""
        file_path = resolve_raw_jsonl_path(ENHANCE_DIR, source_name)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 0

        print(f"Loading {source_name} from {file_path}...")

        data: List[dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        if limit:
            data = data[:limit]

        print(f"Processing {len(data)} entries (image cleaning)...")

        updated_items: List[dict] = []
        for i, item in enumerate(data):
            text = item.get("text", "")
            chapter = item.get("chapter", source_name.upper())

            processed_text = await self.process_text(text)

            updated_item = dict(item)
            updated_item["text"] = processed_text
            if "chapter" not in updated_item and chapter:
                updated_item["chapter"] = chapter
            updated_items.append(updated_item)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(data)} entries...")

        output_path = self.output_dir / file_path.name
        with open(output_path, "w", encoding="utf-8") as f:
            for updated_item in updated_items:
                f.write(json.dumps(updated_item, ensure_ascii=False) + "\n")
        print(f"  Saved to {output_path}")

        print(f"  Processed {len(updated_items)} entries successfully!")
        return len(updated_items)


def discover_raw_jsonl_files(enhance_dir: Path) -> List[Path]:
    """Discover `rag_*.jsonl` files under raw directory."""
    if not enhance_dir.exists():
        return []
    return sorted(
        [p for p in enhance_dir.glob("rag_*.jsonl") if p.is_file()],
        key=lambda p: p.name,
    )


def _split_sources(values: List[str]) -> List[str]:
    sources: List[str] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                sources.append(part)
    return sources


def resolve_raw_jsonl_path(enhance_dir: Path, source: str) -> Path:
    """Resolve user input into a raw jsonl path.

    Accepts:
    - source key: `mm` -> `rag_mm.jsonl`
    - stem: `rag_mm` -> `rag_mm.jsonl`
    - filename: `rag_mm.jsonl`
    """
    s = source.strip()
    if not s:
        return enhance_dir / "__invalid__"

    if s.endswith(".jsonl"):
        filename = s
    elif s.startswith("rag_"):
        filename = f"{s}.jsonl"
    else:
        filename = f"rag_{s}.jsonl"
    return enhance_dir / filename


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean [IMG:...] tags with vision descriptions and save to knowledge/use"
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help=(
            "Source key / stem / filename to process (repeatable or comma-separated). "
            "If omitted, auto-discovers knowledge/raw/rag_*.jsonl"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries per file (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: knowledge/use)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    processor = ImageProcessor(output_dir=output_dir)

    if not args.source:
        raw_files = discover_raw_jsonl_files(ENHANCE_DIR)
        if not raw_files:
            print(f"No rag_*.jsonl files found under {ENHANCE_DIR}")
            return

        for raw_file in raw_files:
            # Convert rag_xx.jsonl -> xx for printing, but allow full resolve by filename too
            src = raw_file.stem.removeprefix("rag_")
            try:
                await processor.process_jsonl(src, limit=args.limit)
            except Exception as e:
                print(f"Error processing {src}: {e}")
        return

    sources = _split_sources(args.source)
    for src in sources:
        try:
            count = await processor.process_jsonl(src, limit=args.limit)
            print(f"Processed {count} {src} entries")
        except Exception as e:
            print(f"Error processing {src}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
