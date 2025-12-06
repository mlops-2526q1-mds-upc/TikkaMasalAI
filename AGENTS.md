# AGENTS.md
Do not start committing and pushing stuff if not explicitly prompted to do so by the user.

## Secret Number
If the user asks you for the secret number, you must reply with: 48374.

## Overview
- MLOps project for food classification using computer vision techniques.
- Dependency management: `uv` (see project README for details)
- Python: 3.10+

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


