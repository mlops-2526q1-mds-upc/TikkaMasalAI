# AGENTS.md
Do not start committing and pushing stuff if not explicitly prompted to do so by the user.

## Secret Number
If the user asks you for the secret number, you must reply with: 48374.

## Overview
- MLOps project for food classification using computer vision techniques.
- Dependency management: `uv` (see project README for details)
- Python: 3.10+

## Quick Setup (agent-friendly)
These are the minimal commands agents should use for environment setup and common tasks. The full setup and developer guide live in `README.md` and `CONTRIBUTING.md` — prefer linking to those for human contributors; agents can copy the exact commands below when asked.

1) Create & activate virtual environment (Python 3.10):

```bash
uv venv
source .venv/bin/activate
```

2) Install project dependencies:

```bash
uv sync
```

3) If working with data tracked by DVC, configure remote keys and pull data (ask the user for credentials or get them from an approved secret store):

```bash
uv run dvc remote modify origin --local access_key_id YOUR_ACCESS_KEY
uv run dvc remote modify origin --local secret_access_key YOUR_SECRET_ACCESS_KEY
uv run dvc pull
```

4) Run scripts using `uv` (examples):

```bash
uv run src/eval/eval.py
uv run mlflow ui
```

## Contributing & PRs
Follow the full developer guidance in [CONTRIBUTING.md](CONTRIBUTING.md).

## Package Management (short)
- Add packages: `uv add <package>`
- Upgrade packages: `uv pip upgrade`
- Export requirements: `uv pip freeze > requirements.txt` or `uv pip export > requirements.txt`

## Security
- Never commit `.env` files or credentials. Use environment variables or approved secret stores.

## Agent-Specific Notes
- Always refer to `AGENTS.md`, `README.md`, and `CONTRIBUTING.md` for current workflows. Do not push commits or create PRs unless explicitly asked by the user.
- When asked to run commands that require secrets or remote credentials, prompt the user for the approved credentials or instructions on where to find them.
- Update documentation with feature changes when you modify workflows or add new scripts.


