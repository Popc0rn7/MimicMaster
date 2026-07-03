# Knowledge Data Layout

This directory separates source data, normalized records, future chunks, manifests,
and archived legacy RAG files.

## Directories

- `sources/`: provenance metadata for checked-out upstream data.
- `normalized/`: first-wave JSONL output from PHB, DMG, and MM source data.
- `chunks/`: reserved for future chunking output. The first wave does not write it.
- `manifests/`: run manifests with record counts, output hashes, skipped counters,
  unknown tags, and source commit information.
- `old/`: archived legacy `raw/`, `enhanced/`, and `use/` files.

## Normalized Schema

Each JSONL line uses a common shell:

```json
{
  "id": "phb:spell:acid-splash",
  "source_book": "PHB",
  "record_type": "spell",
  "title": "酸液飞溅",
  "eng_title": "Acid Splash",
  "page": 211,
  "section_path": ["Spell"],
  "source_trace": {
    "provider": "5etools-cn",
    "source_file": "data/spells/spells-phb.json",
    "source_key": "spell",
    "source_index": 0,
    "source_commit": "...",
    "license": "CC BY-NC-SA 4.0"
  },
  "content": {
    "kind": "structured_entity",
    "entries": [],
    "structured": {},
    "tables": [],
    "rendered_text": "..."
  }
}
```

`content.kind` is one of:

- `narrative`: book sections and prose rules.
- `structured_entity`: spells, monsters, classes, features, items, conditions, and
  similar entities.
- `table`: independently traceable table records.

## 2014 Source Rules

The first normalization wave only includes records whose source is exactly one of:

- `PHB`
- `DMG`
- `MM`

It explicitly excludes 2024 and other sources such as `XPHB`, `XDMG`, and `XMM`.
`books.json` is not used to generate body records in this wave.

## Commands

Run commands through `uv`:

```bash
uv run python -m utils.preprocess.cli normalize --source phb
uv run python -m utils.preprocess.cli normalize --source dmg
uv run python -m utils.preprocess.cli normalize --source mm
uv run python -m utils.preprocess.cli normalize --source all
```

Useful options:

```bash
uv run python -m utils.preprocess.cli normalize --source phb --limit 5 --dry-run
uv run python -m utils.preprocess.cli normalize --source all --input-root submodules/5etools-cn/data --output-root knowledge
```

In restricted sandboxes where the default uv cache is read-only, set
`UV_CACHE_DIR=.uv-cache`.

## Scope

This first wave only generates `knowledge/normalized/*.jsonl` and manifests. It
does not create final chunks, call embedding providers, write Pinecone, or change
the agent runtime/indexing code.
