"""Command line interface for offline knowledge preprocessing."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from utils.preprocess.config import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT
from utils.preprocess.models import BuildManifest, NormalizedRecord
from utils.preprocess.pipelines.dmg import build_dmg_records
from utils.preprocess.pipelines.mm import build_mm_records
from utils.preprocess.pipelines.phb import build_phb_records
from utils.preprocess.processing import process_to_outputs
from utils.preprocess.renderers import RenderStats
from utils.preprocess.sources.fiveetools import FiveEToolsSource
from utils.preprocess.writers import (
    new_timestamp,
    write_jsonl,
    write_manifest,
    write_source_metadata,
)

Pipeline = Callable[[FiveEToolsSource, RenderStats, int | None], object]

PIPELINES: dict[str, Pipeline] = {
    "phb": build_phb_records,
    "dmg": build_dmg_records,
    "mm": build_mm_records,
}

SELECTED_SOURCES: dict[str, list[str]] = {
    "phb": [
        "data/book/book-phb.json",
        "data/spells/spells-phb.json",
        "data/class/class-*.json",
        "data/races.json",
        "data/backgrounds.json",
        "data/feats.json",
        "data/items-base.json",
        "data/items.json",
        "data/conditionsdiseases.json",
        "data/optionalfeatures.json",
        "data/variantrules.json",
    ],
    "dmg": [
        "data/book/book-dmg.json",
        "data/items.json",
        "data/items-base.json",
        "data/rewards.json",
        "data/trapshazards.json",
        "data/objects.json",
        "data/conditionsdiseases.json",
        "data/variantrules.json",
    ],
    "mm": [
        "data/bestiary/bestiary-mm.json",
        "data/bestiary/fluff-bestiary-mm.json",
        "data/book/book-mm.json",
    ],
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "normalize":
        normalize(args)
    elif args.command == "process":
        process(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m utils.preprocess.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument(
        "--source", choices=["phb", "dmg", "mm", "all"], required=True
    )
    normalize_parser.add_argument("--limit", type=int, default=None)
    normalize_parser.add_argument("--dry-run", action="store_true")
    normalize_parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    normalize_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    process_parser = subparsers.add_parser("process")
    process_parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "normalized"
    )
    process_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    process_parser.add_argument("--limit", type=int, default=None)
    return parser


def normalize(args: argparse.Namespace) -> None:
    input_root: Path = args.input_root
    output_root: Path = args.output_root
    selected = ["phb", "dmg", "mm"] if args.source == "all" else [args.source]
    source = FiveEToolsSource(input_root)
    stats = RenderStats()
    outputs: dict[str, dict[str, object]] = {}
    generated: dict[str, list[NormalizedRecord]] = {}

    for source_key in selected:
        records = list(PIPELINES[source_key](source, stats, args.limit))
        generated[source_key] = records
        if args.dry_run:
            outputs[source_key] = {
                "path": str(output_root / "normalized" / f"{source_key}.jsonl"),
                "record_count": len(records),
                "sha256": None,
                "dry_run": True,
            }
        else:
            outputs[source_key] = write_jsonl(
                output_root / "normalized" / f"{source_key}.jsonl", records
            )

    selected_sources = {key: SELECTED_SOURCES[key] for key in selected}
    manifest = BuildManifest(
        generated_at=new_timestamp(),
        source_commit=source.source_commit,
        input_root=str(input_root),
        output_root=str(output_root),
        sources=selected,
        outputs=outputs,
        skipped_items=stats.skipped_items,
        unknown_tags=dict(sorted(stats.unknown_tags.items())),
        empty_records=stats.empty_records,
        source_errors=stats.source_errors,
    )

    if not args.dry_run:
        write_source_metadata(
            output_root,
            source_commit=source.source_commit,
            selected_sources=selected_sources,
            input_root=input_root,
        )
        write_manifest(output_root, manifest)

    print_summary(manifest)


def print_summary(manifest: BuildManifest) -> None:
    print("Normalization summary")
    print(f"  sources: {', '.join(manifest.sources)}")
    for source_key, metadata in manifest.outputs.items():
        print(f"  {source_key}: {metadata['record_count']} records")
    print(f"  skipped_items: {manifest.skipped_items}")
    print(f"  empty_records: {manifest.empty_records}")
    print(f"  unknown_tags: {sum(manifest.unknown_tags.values())}")
    if manifest.source_errors:
        print(f"  source_errors: {len(manifest.source_errors)}")


def process(args: argparse.Namespace) -> None:
    manifest = process_to_outputs(
        normalized_dir=args.input_root,
        output_root=args.output_root,
        limit=args.limit,
    )
    print("Processing summary")
    for key, metadata in manifest.outputs.items():
        print(f"  {key}: {metadata['record_count']} records")
    print(f"  skipped_items: {manifest.skipped_items}")


if __name__ == "__main__":
    main()
