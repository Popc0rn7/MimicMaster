"""Writers for normalized knowledge outputs and manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.preprocess.models import BuildManifest, NormalizedRecord


def write_jsonl(path: Path, records: list[NormalizedRecord]) -> dict[str, Any]:
    """Write records to JSONL and return output metadata."""

    records = ensure_unique_ids(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )
    return {
        "path": str(path),
        "record_count": len(records),
        "sha256": sha256_file(path),
    }


def ensure_unique_ids(records: list[NormalizedRecord]) -> list[NormalizedRecord]:
    """Return records with stable suffixes for duplicate IDs."""

    seen: dict[str, int] = {}
    unique: list[NormalizedRecord] = []
    for record in records:
        count = seen.get(record.id, 0) + 1
        seen[record.id] = count
        if count == 1:
            unique.append(record)
        else:
            unique.append(record.model_copy(update={"id": f"{record.id}:{count}"}))
    return unique


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_manifest(output_root: Path, manifest: BuildManifest) -> Path:
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (
        manifest.generated_at.replace(":", "").replace("-", "").replace("+", "Z")
    )
    manifest_path = manifests_dir / f"{timestamp}.json"
    data = manifest.model_dump(mode="json")
    write_json(manifest_path, data)
    write_json(manifests_dir / "latest.json", data)
    return manifest_path


def write_source_metadata(
    output_root: Path,
    *,
    source_commit: str,
    selected_sources: dict[str, list[str]],
    input_root: Path,
) -> None:
    source_dir = output_root / "sources" / "5etools-cn"
    write_json(
        source_dir / "manifest.json",
        {
            "provider": "5etools-cn",
            "source_commit": source_commit,
            "license": "CC BY-NC-SA 4.0",
            "source_path": str(input_root),
        },
    )
    write_json(source_dir / "selected_sources.json", selected_sources)


def new_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
