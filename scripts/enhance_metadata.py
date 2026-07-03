"""
Metadata enhancement script.

Enhances JSONL files with rich metadata parsed from text content.
Detects data type automatically and extracts appropriate metadata.

Usage:
    python scripts/enhance_metadata.py --source mm
    python scripts/enhance_metadata.py --source phb
    python scripts/enhance_metadata.py --source all

Requirements:
    pip install langchain
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import langchain, fall back to custom implementation if not available
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not installed. Using custom text splitter.")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
ENHANCE_DIR = BASE_DIR / "knowledge" / "enhanced"


_IMG_TAG_PATTERN = re.compile(r"\[IMG:(.*?)\]")


def protect_img_tags(text: str) -> tuple[str, dict[str, str]]:
    """Replace `[IMG:...]` tags with placeholders to avoid being split during chunking.

    Returns:
        protected_text: Text with placeholders.
        mapping: Placeholder -> original tag.
    """
    if not text:
        return text, {}

    mapping: dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:
        original = match.group(0)
        key = f"__IMG_TAG_{len(mapping)}__"
        mapping[key] = original
        # Surround with spaces so splitters can break *around* it safely.
        return f" {key} "

    return _IMG_TAG_PATTERN.sub(_repl, text), mapping


def restore_img_tags(text: str, mapping: dict[str, str]) -> str:
    """Restore placeholders back to original `[IMG:...]` tags."""
    if not text or not mapping:
        return text

    restored = text
    for placeholder, original in mapping.items():
        restored = restored.replace(placeholder, original)
    # Cleanup extra whitespace introduced by placeholders.
    restored = re.sub(r"[ \t]+", " ", restored)
    restored = re.sub(r" \n", "\n", restored)
    restored = re.sub(r"\n ", "\n", restored)
    return restored.strip()


def extract_image_tag(text: str) -> Optional[str]:
    """Extract the full `[IMG:...]` tag from text.

    Note:
        This returns the full tag including brackets, e.g. `[IMG:img/foo.jpg]`.
    """
    match = _IMG_TAG_PATTERN.search(text)
    if not match:
        return None
    # Return the full matched tag, not only the inner path.
    return match.group(0)


def extract_name(text: str) -> str:
    """Extract entry name from text."""
    cleaned = re.sub(r"\[IMG:.*?\]", "", text).strip()
    lines = cleaned.split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("[图片"):
            parts = line.split()
            if parts:
                return parts[0]
    return ""


def extract_monster_stats(text: str) -> Dict[str, Any]:
    """Extract monster statistics from text."""
    result = {}

    # Size mapping
    size_map = {
        "微型": "tiny",
        "小型": "small",
        "中型": "medium",
        "大型": "large",
        "巨型": "huge",
        "超巨型": "gargantuan",
    }

    # Monster types
    monster_types = [
        "aberration",
        "beast",
        "celestial",
        "construct",
        "dragon",
        "elemental",
        "fey",
        "fiend",
        "giant",
        "humanoid",
        "monstrosity",
        "ooze",
        "plant",
        "undead",
        "aarakocra",
        "gnoll",
    ]

    # Ability mapping
    ability_map = {
        "力量": "str",
        "敏捷": "dex",
        "体质": "con",
        "智力": "int",
        "感知": "wis",
        "魅力": "cha",
    }

    # Extract first line
    cleaned = re.sub(r"\[IMG:.*?\]", "", text).strip()
    first_line = ""
    for line in cleaned.split("\n"):
        line = line.strip()
        if line:
            first_line = line
            break

    # Size
    for cn, en in size_map.items():
        if cn in first_line:
            result["size"] = en
            break

    # Type
    for mt in monster_types:
        if mt in first_line.lower():
            result["type"] = mt
            break

    # CR
    cr_match = re.search(r"挑战等级[:\s]*([\d/]+)", text)
    if cr_match:
        result["cr"] = cr_match.group(1)

    # AC
    ac_match = re.search(r"AC[:\s]*(\d+)", text)
    if ac_match:
        result["ac"] = int(ac_match.group(1))

    # HP (only hp_max)
    hp_match = re.search(r"HP[:\s]*(\d+)", text)
    if hp_match:
        result["hp_max"] = int(hp_match.group(1))

    # Speed
    speed = {}
    speed_match = re.search(r"速度[:\s]*(.+?)(?=\s+力量|\s+敏捷|\s+体质|$)", text)
    if speed_match:
        speed_str = speed_match.group(1).strip()
        for cn_type in ["步行", "飞行", "游泳", "攀爬", "挖掘"]:
            if cn_type in speed_str:
                num_match = re.search(r"(\d+)", speed_str)
                if num_match:
                    speed[cn_type] = int(num_match.group(1))
    if speed:
        result["speed"] = speed

    # Abilities
    abilities = {}
    for match in re.finditer(
        r"(力量|敏捷|体质|智力|感知|魅力)(\d+)\(([+-]?\d+)\)", text
    ):
        cn_name = match.group(1)
        en_name = ability_map.get(cn_name, cn_name)
        abilities[en_name] = {"score": int(match.group(2)), "mod": match.group(3)}
    if abilities:
        result["abilities"] = abilities

    # Skills
    skills = []
    skill_match = re.search(
        r"技能[:\s]*(.+?)(?=被动|语言|特性|动作|$)", text, re.DOTALL
    )
    if skill_match:
        for s in re.split(r"[,，]", skill_match.group(1).strip()):
            s = s.strip()
            if s:
                skills.append(s)
    if skills:
        result["skills"] = skills

    # Passive perception
    pp_match = re.search(r"被动感知[:\s]*(\d+)", text)
    if pp_match:
        result["passive_perception"] = int(pp_match.group(1))

    # Languages
    langs = []
    lang_match = re.search(r"语言[:\s]*(.+?)(?=特性|动作|$)", text, re.DOTALL)
    if lang_match:
        for language in re.split(r"[,，]", lang_match.group(1).strip()):
            language = language.strip()
            if language:
                langs.append(language)
    if langs:
        result["languages"] = langs

    # Traits
    traits = []
    trait_match = re.search(r"特性[:\s]*(.+?)(?=动作|$)", text, re.DOTALL)
    if trait_match:
        for t in re.split(r"[。.]", trait_match.group(1).strip()):
            t = t.strip()
            if t:
                traits.append(t)
    if traits:
        result["traits"] = traits

    # Actions
    actions = []
    action_match = re.search(r"动作[:\s]*(.+?)(?=传奇动作|$)", text, re.DOTALL)
    if action_match:
        for a in re.split(r"[。.]", action_match.group(1).strip()):
            a = a.strip()
            if a:
                actions.append(a)
    if actions:
        result["actions"] = actions

    # Legendary actions
    lactions = []
    la_match = re.search(r"传奇动作[:\s]*(.+)$", text, re.DOTALL)
    if la_match:
        for a in re.split(r"[。.]", la_match.group(1).strip()):
            a = a.strip()
            if a:
                lactions.append(a)
    if lactions:
        result["legendary_actions"] = lactions

    return result


def group_by_chapter(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Group all texts by chapter, concatenating in order."""
    groups: Dict[str, List[str]] = defaultdict(list)

    for item in data:
        chapter = item.get("chapter", "")
        text = item.get("text", "")
        if chapter and text:
            groups[chapter].append(text)

    # Join texts with double newline
    return {chapter: "\n".join(texts) for chapter, texts in groups.items()}


def create_custom_text_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    Create a text splitter optimized for D&D rule text.
    Prefers splitting on double newlines (paragraphs), then single newlines,
    then sentences. Avoids breaking numbered list items.
    """
    if not LANGCHAIN_AVAILABLE:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Priority: double newline > single newline > sentence > character
        separators=[
            "\n\n",  # Paragraphs
            "。\n",  # Chinese period + newline
            "\n",  # Single newline
            "。",  # Chinese period
            ". ",  # English period with space
            "；",  # Chinese semicolon
            "; ",  # English semicolon
            "，",  # Chinese comma
            ", ",  # English comma
            " ",  # Space
            "",  # Fallback to character
        ],
        keep_separator=True,
    )


def custom_chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[str]:
    """
    Custom text chunking without langchain.
    Prefers splitting on paragraphs, then sentences, avoiding breaking list items.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    # Try to split by paragraphs first
    paragraphs = re.split(r"\n\n+", text)

    current_chunk = ""
    for para in paragraphs:
        # Check if adding this paragraph would exceed chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Apply overlap by keeping the end of current chunk
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:]
            else:
                current_chunk = ""
            current_chunk += para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

        # If single paragraph exceeds chunk_size, split by sentences
        if len(current_chunk) > chunk_size * 1.5:
            sentences = re.split(r"(?<=[。.!?])\s+", para)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    current_chunk += " " + sent if current_chunk else sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Chunk text using langchain or custom implementation."""
    if LANGCHAIN_AVAILABLE:
        try:
            splitter = create_custom_text_splitter(chunk_size, chunk_overlap)
            if splitter:
                return splitter.split_text(text)
        except Exception as e:
            print(f"  LangChain failed: {e}, using custom splitter")

    return custom_chunk_text(text, chunk_size, chunk_overlap)


def process_non_mm_file(
    file_path: Path,
    output_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """
    Process non-MM JSONL files:
    1. Group texts by chapter
    2. Chunk the combined text
    3. Inject hierarchy path and create output format

    Args:
        file_path: Input file path (from raw directory)
        output_dir: Output directory (use directory)
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
    """
    print(f"Processing {file_path}...")

    # Load raw data (chapter + text format)
    data = load_jsonl(file_path)
    if not data:
        print(f"  No data found in {file_path}")
        return 0

    # Check if already processed (has 'name' field)
    if "name" in data[0]:
        print(f"  File already processed, skipping: {file_path}")
        return 0

    print(f"  Loaded {len(data)} entries, grouping by chapter...")

    # Group by chapter
    chapter_texts = group_by_chapter(data)
    print(f"  Found {len(chapter_texts)} unique chapters")

    # Process each chapter
    results = []
    for chapter, full_text in chapter_texts.items():
        # Parse hierarchy path
        hierarchy_path = chapter.split("/")

        # Protect image tags from being broken by chunking
        protected_text, img_mapping = protect_img_tags(full_text)

        # Chunk the text
        chunks = chunk_text(protected_text, chunk_size, chunk_overlap)

        # Create output entries
        for idx, chunk_text_content in enumerate(chunks):
            chunk_text_content = restore_img_tags(chunk_text_content, img_mapping)
            # Inject path at the beginning of text
            injected_text = f"路径：{chapter}。\n正文：{chunk_text_content}"

            entry = {
                "id": f"{chapter}-{idx}",
                "text": injected_text,
                "metadata": {
                    "hierarchy_path": hierarchy_path,
                    "chunk_index": idx,
                },
            }
            results.append(entry)

    # Save to output directory (use)
    output_path = output_dir / file_path.name
    save_jsonl(results, output_path)
    print(f"  Saved {len(results)} chunks to {output_path}")
    return len(results)


def detect_source_type(text: str) -> str:
    """Detect if text is monster, rule, or other."""
    if re.search(r"挑战等级|AC[:\s]*\d+|HP[:\s]*\d+", text):
        return "monster"
    if "法术" in text or "spell" in text.lower():
        return "spell"
    if "职业" in text or "class" in text.lower():
        return "class_rule"
    if "种族" in text or "race" in text.lower():
        return "race"
    return "rule"


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: Path) -> None:
    """Save data to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def enhance_jsonl(file_path: Path, output_dir: Path) -> int:
    """Enhance entries in a JSONL file (for mm format with monster metadata)."""
    print(f"Loading {file_path}...")
    data = load_jsonl(file_path)

    print(f"Processing {len(data)} entries...")
    enhanced = []

    for i, item in enumerate(data):
        text = item.get("text", "")
        chapter = item.get("chapter", "")

        # Detect type and extract metadata
        data_type = detect_source_type(text)

        if data_type == "monster":
            metadata = extract_monster_stats(text)
            name = extract_name(text)
            image_tag = extract_image_tag(text)
        else:
            # Basic metadata for rules
            section = chapter.split("/")[-1] if "/" in chapter else chapter
            metadata = {"section": chapter}
            name = section
            image_tag = None

        enhanced.append(
            {
                "name": name,
                "image_description": image_tag,
                "metadata": metadata,
                "text": text,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(data)}...")

    # Save to output directory (use)
    output_path = output_dir / file_path.name
    save_jsonl(enhanced, output_path)
    print(f"  Saved {len(enhanced)} entries to {output_path}")
    return len(enhanced)


def main():
    parser = argparse.ArgumentParser(description="Enhance JSONL with rich metadata")
    parser.add_argument(
        "--source",
        choices=["mm", "phb", "dmg", "all"],
        default="all",
        help="Source to process: mm (monster), phb (player's handbook), dmg (DM guide), or all",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for non-MM files (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for non-MM files (default: 50)",
    )
    args = parser.parse_args()

    # Define sources
    if args.source == "all":
        sources = ["mm", "phb", "dmg"]
    else:
        sources = [args.source]

    total = 0

    for src in sources:
        path = ENHANCE_DIR / f"rag_{src}.jsonl"
        if not path.exists():
            # Also check raw directory for unprocessed files
            path = BASE_DIR / "knowledge" / "raw" / f"rag_{src}.jsonl"
            if not path.exists():
                print(f"Skip: {path} not found")
                continue

        if src == "mm":
            # MM uses monster metadata extraction
            total += enhance_jsonl(path, ENHANCE_DIR)
        else:
            # Other sources use chapter grouping + chunking
            total += process_non_mm_file(
                path, ENHANCE_DIR, args.chunk_size, args.chunk_overlap
            )

    print(f"\nTotal: {total} entries/chunks processed")


if __name__ == "__main__":
    main()
