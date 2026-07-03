"""Reader helpers for the checked-out 5etools-cn data tree."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from utils.preprocess.config import PROVIDER, SOURCE_LICENSE
from utils.preprocess.models import SourceTrace


class FiveEToolsSource:
    """Read JSON data from submodules/5etools-cn/data."""

    def __init__(self, input_root: Path) -> None:
        self.input_root = input_root
        self.repo_root = input_root.parent
        self.source_commit = self.get_submodule_commit()

    def read_json(self, relative_path: str) -> dict[str, Any]:
        path = self.input_root / relative_path
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Expected object JSON in {path}")
        return data

    def iter_collection_records(
        self, relative_path: str, source_key: str
    ) -> Iterator[tuple[int, dict[str, Any], SourceTrace]]:
        data = self.read_json(relative_path)
        records = data.get(source_key, [])
        if not isinstance(records, list):
            raise ValueError(f"Expected list at {relative_path}:{source_key}")
        for index, record in enumerate(records):
            if isinstance(record, dict):
                yield index, record, self.trace(relative_path, source_key, index)

    def iter_book_sections(
        self, relative_path: str
    ) -> Iterator[tuple[int, dict[str, Any], SourceTrace]]:
        yield from self.iter_collection_records(relative_path, "data")

    def class_files(self) -> list[str]:
        class_dir = self.input_root / "class"
        return sorted(
            str(path.relative_to(self.input_root))
            for path in class_dir.glob("class-*.json")
        )

    def trace(
        self, relative_path: str, source_key: str, source_index: int | None
    ) -> SourceTrace:
        return SourceTrace(
            provider=PROVIDER,
            source_file=f"data/{relative_path}",
            source_key=source_key,
            source_index=source_index,
            source_commit=self.source_commit,
            license=SOURCE_LICENSE,
        )

    def get_submodule_commit(self) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return result.stdout.strip() or "unknown"
