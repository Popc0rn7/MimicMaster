# Tests

Default tests are offline regression tests. They use fake local configuration from
`tests/conftest.py` and should not call external providers.

## E2E Smoke Tests

All real-service smoke tests use the same opt-in switch:

```bash
RUN_E2E=1 uv run pytest tests/e2e -v
```

Run one service at a time when you are checking a specific dependency:

```bash
RUN_E2E=1 uv run pytest tests/e2e/test_embedding_service.py -v
RUN_E2E=1 uv run pytest tests/e2e/test_pinecone_service.py -v
RUN_E2E=1 uv run pytest tests/e2e/test_mongodb_service.py -v
RUN_E2E=1 uv run pytest tests/e2e/test_agent_service.py -v
RUN_E2E=1 uv run pytest tests/e2e/test_llm_provider.py -v
RUN_E2E=1 uv run pytest tests/e2e/test_vision_model.py -v
```

`uv run --no-sync ...` is optional. It skips dependency synchronization and is
useful when the virtual environment is already correct or when the sandbox blocks
package/build writes. You can omit it in normal local development.

The Agent API smoke test currently verifies that the endpoint can process a
request. The current `DMAgent` still returns a placeholder response. Set
`REQUIRE_REAL_AGENT_LLM=1` when a real LLM-backed agent is implemented and you
want the smoke test to reject placeholder responses.

The generic LLM provider smoke test uses an OpenAI-compatible chat-completions
interface. Set `LLM_PROVIDER_API_KEY`, `LLM_PROVIDER_BASE_URL`, and
`LLM_PROVIDER_MODEL` when you want to run it.
