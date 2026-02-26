"""
Knowledge indexing script.

Indexes D&D knowledge from JSONL files into Pinecone with proper
namespace and metadata organization.

Usage:
    python scripts/index_knowledge.py --source mm --limit 10
    python scripts/index_knowledge.py --source phb
    python scripts/index_knowledge.py --source all
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


# Base paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "knowledge" / "raw"


class KnowledgeIndexer:
    """Indexer for D&D knowledge base."""

    def __init__(self, batch_size: int = 32):
        self.retriever = get_hybrid_knowledge_retriever()
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
            # Replace with placeholder for now
            processed = re.sub(pattern, "[图片占位符]", text)
            return processed, image_path
        return text, None

    def parse_monster_data(self, text: str, chapter: str) -> KnowledgeMetadata:
        """Parse monster data from text."""
        metadata = KnowledgeMetadata(
            category=KnowledgeCategory.MONSTER,
            source_book=SourceBook.MM,
            chapter=chapter,
        )

        # Try to extract monster name (usually at the beginning)
        # Format: "[IMG:...] Name Type Size Alignment CR"
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

    def parse_rule_data(self, text: str, chapter: str, source: str) -> KnowledgeMetadata:
        """Parse rule/class/spell data from text."""
        metadata = KnowledgeMetadata(
            category=KnowledgeCategory.RULE,
            source_book=source,
            chapter=chapter,
        )

        # Try to detect rule type from chapter or text
        text_lower = text.lower()
        if "法术" in text or "spell" in text_lower:
            metadata.rule_type = "spell"
        elif "职业" in text or "class" in text_lower:
            metadata.rule_type = "class"
        elif "种族" in text or "race" in text_lower:
            metadata.rule_type = "race"
        elif "背景" in text or "background" in text_lower:
            metadata.rule_type = "background"
        elif "专长" in text or "feat" in text_lower:
            metadata.rule_type = "feat"
        elif "技能" in text or "skill" in text_lower:
            metadata.rule_type = "skill"
        elif "装备" in text or "equipment" in text_lower:
            metadata.rule_type = "equipment"

        return metadata

    def load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    async def index_monsters(self, limit: Optional[int] = None) -> int:
        """Index monster data from rag_mm.jsonl."""
        file_path = RAW_DIR / "rag_mm.jsonl"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 0

        print(f"Loading monsters from {file_path}...")
        data = self.load_jsonl(file_path)

        if limit:
            data = data[:limit]

        print(f"Indexing {len(data)} monsters to 'monsters' namespace...")

        ids = []
        texts = []
        metadata_list = []

        for i, item in enumerate(data):
            text = item.get("text", "")
            chapter = item.get("chapter", "MM")

            # Process image placeholder
            processed_text, image_path = self.detect_image_in_text(text)
            if image_path:
                processed_text = processed_text.replace(
                    "[图片占位符]",
                    f"[图片: {image_path}]"
                )

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

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(data)} monsters...")

        # Batch index
        print(f"  Upserting to Pinecone...")
        await self.retriever.index_documents(
            ids=ids,
            texts=texts,
            metadata=metadata_list,
            namespace=settings.monsters_namespace,
        )

        print(f"  Indexed {len(ids)} monsters successfully!")
        return len(ids)

    async def index_rules(self, source: str = "phb", limit: Optional[int] = None) -> int:
        """Index rules from PHB or DMG."""
        if source == "phb":
            file_path = RAW_DIR / "rag_phb.jsonl"
            source_book = SourceBook.PHB
        elif source == "dmg":
            file_path = RAW_DIR / "rag_dmg.jsonl"
            source_book = SourceBook.DMG
        else:
            print(f"Unknown source: {source}")
            return 0

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 0

        print(f"Loading {source.upper()} rules from {file_path}...")
        data = self.load_jsonl(file_path)

        if limit:
            data = data[:limit]

        namespace = settings.rules_namespace
        print(f"Indexing {len(data)} entries to '{namespace}' namespace...")

        ids = []
        texts = []
        metadata_list = []

        for i, item in enumerate(data):
            text = item.get("text", "")
            chapter = item.get("chapter", source.upper())

            # Parse metadata
            metadata = self.parse_rule_data(text, chapter, source_book)

            # Generate ID
            doc_id = f"{source}_{i:04d}"

            ids.append(doc_id)
            texts.append(text)
            metadata_list.append(metadata.to_dict())

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(data)} entries...")

        # Batch index
        print(f"  Upserting to Pinecone...")
        await self.retriever.index_documents(
            ids=ids,
            texts=texts,
            metadata=metadata_list,
            namespace=namespace,
        )

        print(f"  Indexed {len(ids)} {source.upper()} entries successfully!")
        return len(ids)

    async def index_all(self, limit: Optional[int] = None) -> Dict[str, int]:
        """Index all knowledge sources."""
        results = {}

        print("\n" + "=" * 50)
        print("Starting full knowledge indexing...")
        print("=" * 50 + "\n")

        # Index monsters
        results["monsters"] = await self.index_monsters(limit=limit)

        # Index PHB rules
        results["phb"] = await self.index_rules("phb", limit=limit)

        # Index DMG rules
        results["dmg"] = await self.index_rules("dmg", limit=limit)

        print("\n" + "=" * 50)
        print("Indexing complete!")
        print("=" * 50)
        print(f"Results:")
        print(f"  - Monsters: {results.get('monsters', 0)}")
        print(f"  - PHB Rules: {results.get('phb', 0)}")
        print(f"  - DMG Rules: {results.get('dmg', 0)}")
        print(f"  - Total: {sum(results.values())}")

        return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index D&D knowledge to Pinecone")
    parser.add_argument(
        "--source",
        choices=["mm", "phb", "dmg", "all"],
        default="all",
        help="Source to index (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to index (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for indexing",
    )

    args = parser.parse_args()

    indexer = KnowledgeIndexer(batch_size=args.batch_size)

    if args.source == "all":
        await indexer.index_all(limit=args.limit)
    elif args.source == "mm":
        count = await indexer.index_monsters(limit=args.limit)
        print(f"Indexed {count} monsters")
    elif args.source == "phb":
        count = await indexer.index_rules("phb", limit=args.limit)
        print(f"Indexed {count} PHB entries")
    elif args.source == "dmg":
        count = await indexer.index_rules("dmg", limit=args.limit)
        print(f"Indexed {count} DMG entries")


if __name__ == "__main__":
    asyncio.run(main())
