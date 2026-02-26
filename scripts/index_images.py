"""
Image indexing script.

Processes images in knowledge base using vision model (GLM-4V),
updates JSONL files with descriptions, and indexes to Pinecone.

Usage:
    python scripts/index_images.py --source mm --limit 10
    python scripts/index_images.py --source mm
    python scripts/index_images.py --source dmg
    python scripts/index_images.py --source all
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mimic_master.config import settings
from mimic_master.memory.knowledge_retriever import get_hybrid_knowledge_retriever
from mimic_master.models.memory import (
    KnowledgeCategory,
    KnowledgeMetadata,
    MONSTER_TYPES,
    SourceBook,
)
from mimic_master.services.vision_service import get_vision_service


# Base paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "knowledge" / "raw"


class ImageIndexer:
    """Indexer for processing images with vision model."""

    def __init__(self, batch_size: int = 10):
        self.retriever = get_hybrid_knowledge_retriever()
        self.vision_service = get_vision_service()
        self.batch_size = batch_size

    def detect_image_in_text(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Detect [IMG:xxx] pattern in text.

        Returns:
            Tuple of (processed_text, image_path or None)
        """
        pattern = r"\[IMG:(.*?)\]"
        match = re.search(pattern, text)
        if match:
            image_path = match.group(1)
            # Remove the IMG tag for processing
            processed = re.sub(pattern, "", text).strip()
            return processed, image_path
        return text, None

    async def process_image(self, image_path: str) -> str:
        """
        Process a single image and return description.

        Args:
            image_path: Path to the image

        Returns:
            Image description text
        """
        try:
            result = await self.vision_service.describe_image(image_path)
            return f"[图片描述: {result.description}]"
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            return f"[图片: {image_path}]"

    async def index_monsters(self, limit: Optional[int] = None, dry_run: bool = False) -> int:
        """Index monster images from rag_mm.jsonl."""
        file_path = RAW_DIR / "rag_mm.jsonl"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 0

        print(f"Loading monsters from {file_path}...")

        # Load existing data
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        if limit:
            data = data[:limit]

        print(f"Processing {len(data)} monsters with vision model...")

        ids = []
        texts = []
        metadata_list = []

        for i, item in enumerate(data):
            text = item.get("text", "")
            chapter = item.get("chapter", "MM")

            # Detect image in text
            processed_text, image_path = self.detect_image_in_text(text)

            if image_path:
                # Get image description from vision model
                image_description = await self.process_image(image_path)
                # Prepend description to text
                processed_text = f"{image_description}\n{processed_text}"

            # Parse metadata
            metadata = self.parse_monster_data(text, chapter)
            if image_path:
                metadata.has_image = True
                metadata.image_path = image_path

            # Generate ID
            doc_id = f"monster_{i:04d}"

            ids.append(doc_id)
            texts.append(processed_text)
            metadata_list.append(metadata.to_dict())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(data)} monsters...")

        # Save updated JSONL if not dry run
        if not dry_run:
            output_path = RAW_DIR / "rag_mm_described.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            print(f"  Saved updated JSONL to {output_path}")

        # Index to Pinecone
        print(f"  Upserting to Pinecone...")
        await self.retriever.index_documents(
            ids=ids,
            texts=texts,
            metadata=metadata_list,
            namespace=settings.monsters_namespace,
        )

        print(f"  Indexed {len(ids)} monsters with image descriptions successfully!")
        return len(ids)

    async def index_dmg(self, limit: Optional[int] = None, dry_run: bool = False) -> int:
        """Index DMG images from rag_dmg.jsonl."""
        file_path = RAW_DIR / "rag_dmg.jsonl"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 0

        print(f"Loading DMG entries from {file_path}...")

        # Load existing data
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        if limit:
            data = data[:limit]

        print(f"Processing {len(data)} DMG entries with vision model...")

        ids = []
        texts = []
        metadata_list = []

        for i, item in enumerate(data):
            text = item.get("text", "")
            chapter = item.get("chapter", "DMG")

            # Detect image in text
            processed_text, image_path = self.detect_image_in_text(text)

            if image_path:
                # Get image description from vision model
                image_description = await self.process_image(image_path)
                # Prepend description to text
                processed_text = f"{image_description}\n{processed_text}"

            # Parse metadata
            metadata = KnowledgeMetadata(
                category=KnowledgeCategory.RULE,
                source_book=SourceBook.DMG,
                chapter=chapter,
            )
            if image_path:
                metadata.has_image = True
                metadata.image_path = image_path

            # Generate ID
            doc_id = f"dmg_{i:04d}"

            ids.append(doc_id)
            texts.append(processed_text)
            metadata_list.append(metadata.to_dict())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(data)} entries...")

        # Save updated JSONL if not dry run
        if not dry_run:
            output_path = RAW_DIR / "rag_dmg_described.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            print(f"  Saved updated JSONL to {output_path}")

        # Index to Pinecone
        print(f"  Upserting to Pinecone...")
        await self.retriever.index_documents(
            ids=ids,
            texts=texts,
            metadata=metadata_list,
            namespace=settings.rules_namespace,
        )

        print(f"  Indexed {len(ids)} DMG entries with image descriptions successfully!")
        return len(ids)

    def parse_monster_data(self, text: str, chapter: str) -> KnowledgeMetadata:
        """Parse monster data from text."""
        metadata = KnowledgeMetadata(
            category=KnowledgeCategory.MONSTER,
            source_book=SourceBook.MM,
            chapter=chapter,
        )

        # Extract monster name
        lines = text.split("\n")
        if lines:
            first_line = lines[0]
            # Remove IMG tag
            first_line = re.sub(r"\[IMG:.*?\]", "", first_line).strip()
            parts = first_line.split()

            if parts:
                metadata.name = parts[0]

                # Try to find monster type
                for mt in MONSTER_TYPES:
                    if mt in first_line.lower():
                        metadata.monster_type = mt
                        break

                # Try to find CR
                cr_match = re.search(r"CR[:\s]*(\d+/\d+|\d+)", first_line, re.IGNORECASE)
                if cr_match:
                    metadata.cr = cr_match.group(1)

        # Check for image
        _, image_path = self.detect_image_in_text(text)
        if image_path:
            metadata.has_image = True
            metadata.image_path = image_path

        return metadata

    async def index_all(self, limit: Optional[int] = None, dry_run: bool = False) -> Dict[str, int]:
        """Index all images."""
        results = {}

        print("\n" + "=" * 50)
        print("Starting full image indexing with GLM-4V...")
        print("=" * 50 + "\n")

        # Index monsters
        results["monsters"] = await self.index_monsters(limit=limit, dry_run=dry_run)

        # Index DMG
        results["dmg"] = await self.index_dmg(limit=limit, dry_run=dry_run)

        print("\n" + "=" * 50)
        print("Image indexing complete!")
        print("=" * 50)
        print(f"Results:")
        print(f"  - Monsters: {results.get('monsters', 0)}")
        print(f"  - DMG: {results.get('dmg', 0)}")
        print(f"  - Total: {sum(results.values())}")

        return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process images with vision model and index to Pinecone"
    )
    parser.add_argument(
        "--source",
        choices=["mm", "dmg", "all"],
        default="all",
        help="Source to process (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process images but don't save to JSONL or Pinecone",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    indexer = ImageIndexer(batch_size=args.batch_size)

    if args.source == "all":
        await indexer.index_all(limit=args.limit, dry_run=args.dry_run)
    elif args.source == "mm":
        count = await indexer.index_monsters(limit=args.limit, dry_run=args.dry_run)
        print(f"Processed {count} monsters")
    elif args.source == "dmg":
        count = await indexer.index_dmg(limit=args.limit, dry_run=args.dry_run)
        print(f"Processed {count} DMG entries")


if __name__ == "__main__":
    asyncio.run(main())
