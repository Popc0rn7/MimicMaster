# Tests

Default tests are offline regression tests. They use fake local configuration from
`tests/conftest.py` and should not call external providers.

## Service Smoke Tests

Use these when you want one command to prove the configured services are wired
end to end.

Run both embedding and Agent API smoke tests:

```bash
RUN_SERVICE_E2E=1 UV_CACHE_DIR=.uv-cache uv run --no-sync pytest tests/e2e/test_services_e2e.py -v
```

Run only the embedding service smoke test:

```bash
RUN_REAL_EMBEDDING=1 UV_CACHE_DIR=.uv-cache uv run --no-sync pytest tests/e2e/test_services_e2e.py::test_embedding_service_end_to_end_current_config -v
```

Run only the Agent API smoke test:

```bash
RUN_AGENT_E2E=1 UV_CACHE_DIR=.uv-cache uv run --no-sync pytest tests/e2e/test_services_e2e.py::test_agent_api_end_to_end_current_app -v
```

The Agent API smoke test verifies the application endpoint can process a request.
The current `DMAgent` still returns a placeholder response. Set
`REQUIRE_REAL_AGENT_LLM=1` when a real LLM-backed agent is implemented and you
want the smoke test to reject placeholder responses.
