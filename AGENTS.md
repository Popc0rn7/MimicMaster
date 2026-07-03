# Repository Guidelines

## Project Structure & Module Organization

Mimic Master is a Python 3.12 FastAPI service for a D&D 5E DM assistant. Core code lives in `mimic_master/`: `api/` exposes routes, `core/` coordinates agent behavior, `memory/` implements state, knowledge, and episodic memory, `models/` contains Pydantic schemas, `services/` wraps external providers, and `utils/` holds helpers. `main.py` is the local entry point. Tests are in `tests/`, scripts are in `scripts/`, docs are in `docs/`, and JSONL knowledge assets are under `knowledge/`.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `python main.py`: start the API using the repository entry point.
- `uv run uvicorn mimic_master.api.app:app --reload`: run the FastAPI app with reload for development.
- `uv run pytest tests/ -v`: run the full test suite with verbose output.
- `uv run ruff check .`: lint Python files.
- `uv run black --check .`: verify formatting without modifying files.
- `uv run python scripts/index_knowledge.py --source mm --limit 10`: test knowledge indexing on a small sample.

## Coding Style & Naming Conventions

Use type hints for all new Python functions and methods. Follow Black formatting and Ruff lint rules; CI enforces both. Prefer clear, lightweight modules over deep abstraction. Use snake_case for functions, variables, files, and tests; use PascalCase for classes and Pydantic models. Keep environment names consistent with `NAMING.md`, such as `mimic-rules-dev` indexes and namespaces like `rules`, `episodes`, `monsters`, and `spells`.

## Testing Guidelines

Pytest is the test framework, with `pytest-asyncio` for async code. Name files `tests/test_*.py` and test functions `test_*`. Use monkeypatch/fake services for NVIDIA embedding, reranker, Pinecone, and MongoDB calls where possible; some retrieval tests require configured external indexes. For local coverage, use `tests/run_tests.sh`.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add vision service` or `Compatible offline reranker and embedding`. Keep commits focused and mention affected areas when useful. Pull requests should include a concise summary, test results, linked issues when applicable, and API examples or screenshots for user-facing behavior. Note any required `.env` keys, external services, or intentionally skipped tests.

## Security & Configuration Tips

Do not commit secrets. Copy `.env.example` to `.env` for local settings, and keep provider URLs, API keys, LangSmith tracing, Pinecone indexes, and MongoDB configuration in environment variables. Prefer mock provider settings for local tests unless validating integrations.
