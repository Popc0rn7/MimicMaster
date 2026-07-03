# External Knowledge Sources

Use this directory to record or temporarily place source material that is not
available in the local checkout.

Do not commit copyrighted source text or large generated datasets here. Keep
external originals outside git, then import or convert them into:

- `knowledge/raw/` for source JSONL files
- `knowledge/use/` for processed JSONL files used by `scripts/index_knowledge.py`

Expected processed filenames:

- `knowledge/use/rag_mm.jsonl`
- `knowledge/use/rag_phb.jsonl`
- `knowledge/use/rag_dmg.jsonl`

If a processed file is missing or empty, restore the source data here or in
external storage first, then rebuild the processed file before indexing.
