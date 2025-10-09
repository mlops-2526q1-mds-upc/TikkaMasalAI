# Contributing to TikkaMasalAI

Thanks for your interest in contributing! This document explains how to propose changes, the pull request (PR) workflow, coding standards, and tooling (Ruff, tests, pre-commit).

## Table of Contents
- [Getting Started](#getting-started)
- [Branch Strategy & Naming](#branch-strategy--naming)
- [Commit Messages](#commit-messages)
- [Pull Request Checklist](#pull-request-checklist)
- [Code Style & Linting (Ruff)](#code-style--linting-ruff)
- [Formatting & Pre-commit Hooks](#formatting--pre-commit-hooks)
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

# install pre-commit hooks
pre-commit install
pre-commit run --all-files  # initial cleanup
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

Required: prepend the corresponding emoji at the start of PR titles to make intent obvious in the review list. Example PR title formats (mandatory):

```
✨ feat: add food101 evaluator sampling
🐛 fix: correct resnet18 loading bug
📝 docs: update contributing guide with emojis
```

These emojis are mandatory for PR titles. The PR checklist and reviewers may ask you to update titles that don't follow this convention; CI or bots may also enforce this in the future.

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
- [ ] `pre-commit run --all-files` clean / staged any modifications
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

## Formatting & Pre-commit Hooks
We use a single Ruff pre-commit hook (auto-fix). Install once:
```bash
pip install pre-commit
pre-commit install
```
On commit, Ruff may modify files; re-stage and re-commit if needed.
Manual full pass (recommended before PR):
```bash
make format          # ruff check --fix + ruff format
pre-commit run --all-files
```

## Testing Guidelines
- Put tests under `tests/`.
- Prefer deterministic tests; control randomness (`seed=42` etc.).
- Add at least one regression test for each bug fix.
- Keep unit tests fast (<1s each when possible) so CI stays responsive.
- Use smoke tests for integration-heavy paths (models, dataset loading) to ensure import + minimal inference path works.

Run tests:
```bash
uv run pytest -q
```

## Documentation Expectations
Update or add:
- README sections if public interface changes
- Model cards (in `modelcard_*.md`) for model-related updates
- `docs/` pages if user workflows change
- Inline docstrings for public classes/functions

## CI Pipeline
GitHub Actions workflows:
- `ruff.yml` – lint on every push + PR
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