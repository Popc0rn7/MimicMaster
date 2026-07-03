"""Small 5etools JSON entry renderer for normalized text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

KNOWN_INLINE_TAGS = {
    "b",
    "i",
    "spell",
    "item",
    "creature",
    "damage",
    "dc",
    "skill",
    "status",
    "condition",
    "chance",
    "dice",
    "hit",
    "h",
    "atk",
    "filter",
    "book",
    "adventure",
    "sense",
    "action",
    "race",
    "background",
    "class",
    "feat",
    "recharge",
}
TAG_RE = re.compile(r"\{@([a-zA-Z0-9_]+)\s+([^{}]*)\}")


@dataclass
class RenderStats:
    """Mutable counters collected while rendering source entries."""

    unknown_tags: dict[str, int] = field(default_factory=dict)
    skipped_items: int = 0
    empty_records: int = 0
    source_errors: list[str] = field(default_factory=list)

    def count_unknown(self, tag: str) -> None:
        self.unknown_tags[tag] = self.unknown_tags.get(tag, 0) + 1


def render_inline(value: Any, stats: RenderStats | None = None) -> str:
    """Render inline strings and 5etools tags to readable text."""

    if value is None:
        return ""
    text = str(value)

    def replace(match: re.Match[str]) -> str:
        tag = match.group(1)
        payload = match.group(2)
        display = payload.split("|", 1)[0]
        if tag == "dc":
            display = f"DC {display}"
        elif tag == "hit":
            display = f"+{display.lstrip('+')}"
        elif tag == "h":
            display = ""
        elif tag == "atk":
            display = display.replace("mw", "melee weapon").replace(
                "rw", "ranged weapon"
            )
        elif tag == "recharge":
            display = f"Recharge {display}"
        if stats is not None and tag not in KNOWN_INLINE_TAGS:
            stats.count_unknown(tag)
        return display

    previous = None
    while previous != text:
        previous = text
        text = TAG_RE.sub(replace, text)
    return re.sub(r"\s+", " ", text).strip()


def render_entries(entries: Any, stats: RenderStats | None = None) -> str:
    """Render nested 5etools entries into stable readable plain text."""

    lines = _render_entry(entries, stats)
    return "\n".join(line for line in lines if line.strip()).strip()


def extract_tables(
    entries: Any, stats: RenderStats | None = None
) -> list[dict[str, Any]]:
    """Return rendered table structures contained in nested entries."""

    tables: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "table":
            tables.append(normalize_table(node, stats))
            return
        for value in node.values():
            walk(value)

    walk(entries)
    return tables


def normalize_table(
    table: dict[str, Any], stats: RenderStats | None = None
) -> dict[str, Any]:
    """Normalize a 5etools table node without losing row structure."""

    labels = [render_inline(label, stats) for label in table.get("colLabels", [])]
    rows = []
    for row in table.get("rows", []):
        if isinstance(row, list):
            rows.append([render_inline(cell, stats) for cell in row])
        else:
            rows.append([render_inline(row, stats)])
    return {
        "caption": render_inline(table.get("caption", ""), stats),
        "col_labels": labels,
        "rows": rows,
    }


def _render_entry(entry: Any, stats: RenderStats | None) -> list[str]:
    if entry is None:
        return []
    if isinstance(entry, str):
        rendered = render_inline(entry, stats)
        return [rendered] if rendered else []
    if isinstance(entry, (int, float, bool)):
        return [str(entry)]
    if isinstance(entry, list):
        lines: list[str] = []
        for item in entry:
            lines.extend(_render_entry(item, stats))
        return lines
    if not isinstance(entry, dict):
        return [render_inline(entry, stats)]

    entry_type = entry.get("type")
    if entry_type in {"image", "internal", "gallery"}:
        if stats is not None:
            stats.skipped_items += 1
        return []
    if entry_type == "table":
        table = normalize_table(entry, stats)
        lines = [table["caption"]] if table["caption"] else []
        if table["col_labels"]:
            lines.append(" | ".join(table["col_labels"]))
        lines.extend(" | ".join(row) for row in table["rows"])
        return lines
    if entry_type in {"list", "options"}:
        lines = []
        for item in entry.get("items", entry.get("entries", [])):
            rendered = render_entries(item, stats)
            if rendered:
                lines.append(f"- {rendered}")
        return lines
    if entry_type in {"entries", "section", "inset", "quote", "variant"}:
        lines = []
        name = render_inline(entry.get("name", ""), stats)
        if name:
            lines.append(name)
        lines.extend(_render_entry(entry.get("entries", []), stats))
        return lines
    if entry_type == "refOptionalfeature":
        return [render_inline(entry.get("optionalfeature", ""), stats)]

    lines = []
    name = render_inline(entry.get("name", ""), stats)
    if name:
        lines.append(name)
    if "entries" in entry:
        lines.extend(_render_entry(entry["entries"], stats))
    elif "items" in entry:
        lines.extend(_render_entry(entry["items"], stats))
    return lines
