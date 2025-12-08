# Contributing to TikkaMasalAI

Thanks for your interest in contributing! This document explains how to propose changes, the pull request (PR) workflow, coding standards, and tooling (Ruff, tests).

## Table of Contents
- [Getting Started](#getting-started)
- [Branch Strategy & Naming](#branch-strategy--naming)
- [Commit Messages](#commit-messages)
- [Pull Request Checklist](#pull-request-checklist)
- [Code Style & Linting (Ruff)](#code-style--linting-ruff)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Expectations](#documentation-expectations)
- [CI Pipeline](#ci-pipeline)
- [Dependency Management](#dependency-management)
- [Performance Considerations](#performance-considerations)
- [Security & Secrets](#security--secrets)

## Getting Started
```bash
# clone
git clone https://github.com/mlops-2526q1-mds-upc/TikkaMasalAI.git
cd TikkaMasalAI

# create environment (uv)
uv venv --python 3.10
source .venv/bin/activate

# install dependencies
uv sync
```
> If you use another virtual env tool (e.g. conda, pyenv), ensure Python matches `>=3.10,<3.13`.

## Branch Strategy & Naming
All work happens on feature branches, then merged via PR into `main`.
Suggested prefixes:
-- `feat/<short-description>` – new features  ✨
-- `fix/<bug-id-or-short>` – bug fixes  🐛
-- `refactor/<area>` – internal code changes without behavior change  ♻️
-- `docs/<area>` – documentation only  📝
-- `chore/<task>` – tooling, config, CI  🔧
-- `exp/<experiment>` – one-off experiments (usually not merged without cleanup)  🧪

Examples:
```
feat/food101-evaluator-sampling
fix/resnet18-loading-bug
chore/ruff-ci-all-branches
```

Pull request titles should start with the same short prefix to communicate intent quickly. Example title formats:

```
feat: add food101 evaluator sampling
fix: correct resnet18 loading bug
docs: update contributing guide with new testing section
```

## Commit Messages
Follow conventional-ish style:
```
<type>(optional scope): <imperative summary>

<body (optional)>
```
Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.
Keep summary <= 72 chars. Body (if present) explains what/why, not how.

## Pull Request Checklist
Before opening / marking ready for review:
- [ ] Branch up to date with latest `main`
- [ ] `make format` run (or `ruff check --fix && ruff format`)
- [ ] `ruff check` passes with no errors
- [ ] Tests added/updated (if behavior change)
- [ ] `uv run pytest -q` passes locally
- [ ] No large unrelated refactors
- [ ] Updated docs / README / model cards if user-facing changes
- [ ] No secrets / credentials included

Small PRs (< ~400 lines diff) get reviewed faster. Split large changes if possible.

## Code Style & Linting (Ruff)
We use [Ruff](https://docs.astral.sh/ruff/) for:
- Lint rules (PEP8, unused imports, etc.)
- Import sorting (rule `I` enabled)
- Formatting (`ruff format`)

Configuration lives in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 99
src = ["src", "tests"]

[tool.ruff.lint]
extend-select = ["I"]
per-file-ignores = { "tests/**" = ["D101", "D102", "D103"] }
```
Core commands:
```bash
ruff check .          # report issues
ruff check --fix .    # auto-fix where possible
ruff format           # apply Ruff's formatter
```
CI runs `ruff check .` on every push + PR.

Full code style (naming, docstrings, typing, imports, tests, performance, security) lives in `docs/development/code_style.md`. Treat that document as the single source of truth. Propose any style changes by updating that file in the same PR.

## Testing Guidelines
- Put tests under `tests/`.
- Prefer deterministic tests; control randomness (`seed=42` etc.).
- Add at least one regression test for each bug fix.
- Keep unit tests fast (<1s each when possible) so CI stays responsive.
- Use smoke tests for integration-heavy paths (models, dataset loading) to ensure import + minimal inference path works.

Run tests:
```bash
make test            # preferred (uses uv in Makefile)
# or
uv run pytest -q     # direct invocation
```

API smoke tests (Bruno):
```bash
make test-local-api      # hits local docker-compose stack
make test-deployed-api   # hits the remote environment defined in tikkamasalai-requests/environments/production.bru
```
> Configure the Bruno environment files with the correct base URLs + secrets before running the deployed variant.

## Documentation Expectations
Update or add:
- README sections if public interface changes
- Model cards (in `modelcard_*.md`) for model-related updates
- `docs/` pages if user workflows change
- Inline docstrings for public classes/functions

### Backend API docs (OpenAPI/Redoc)
- When you change backend endpoints or request/response models, regenerate the API docs.
- Prerequisites: Node.js ≥ 20 (for Redoc CLI).
- Commands:
	- `make api-docs` to refresh `src/backend/openapi.json` and `docs/docs/api.html`.
	- `make docs-build` to validate the docs site builds.
- CI will fail if `docs/docs/api.html` is stale; commit regenerated files.

## CI Pipeline
GitHub Actions workflows:
- `ruff.yml` – lint on every push + PR
- `docs.yml` – builds backend API docs and MkDocs; fails if `docs/docs/api.html` is out of date.
Future additions may include: tests, build, deployment. PRs should keep lint green.

## Dependency Management
Managed via `pyproject.toml` + `uv.lock`.
- Add runtime deps to `dependencies`.
- Dev-only tools (if re-introduced) go under `[dependency-groups].dev`.
- Prefer pinning heavy ML libs to known-good versions.
- Remove unused dependencies proactively.

## Performance Considerations
When changing model or data code:
- Avoid unnecessary tensor/device transfers
- Batch operations where possible
- Log rough timing locally if a change might slow training/eval

## Security & Secrets
- Never commit API keys, tokens, dataset credentials
- Use environment variables (`.env` not committed) or secret stores
- Treat model artifact paths as potentially sensitive