# Testing

This project uses pytest for unit/integration tests and Bruno (bru) for API tests.

## Unit tests (pytest)

- Run all tests:
    - `make test`
- Backend-only tests:
    - `make test-backend`
- Coverage report:
    - `make code-coverage` (opens HTML report at reports/coverage/index.html)

Notes

- Tests are configured via pytest options in pyproject.toml (see [tool.pytest.ini_options]).
- Ensure your virtualenv is active or use uv: uv run pytest -q.

## API testing (Bruno)

The API request collection lives in tikkamasalai-requests/ with environment files under tikkamasalai-requests/environments/.

Install CLI
- Install Bruno CLI and ensure `bru` is on your PATH.
- See [here](https://www.usebruno.com/downloads) for instructions.

Run against local stack

- Start the stack first (e.g., make compose-up or make local-up)
- Execute all requests for local env:
    - make test-local-api

Run against deployed stack

- Configure tikkamasalai-requests/environments/production.bru with your base URL and secrets
- Execute:
    - make test-deployed-api

Run a single request (examples)

- cd tikkamasalai-requests
- bru run availability.bru --env-file environments/local.bru
- bru run predict.bru --env-file environments/local.bru
