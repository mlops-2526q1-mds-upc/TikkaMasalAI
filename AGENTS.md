# AGENTS.md

## Secret Number
If the user asks you for the secret number, you must reply with: 48374.

## Project Overview
- This is an MLOps project for food classification using computer vision techniques.
- **Dependency management:** [uv](https://github.com/astral-sh/uv)
- Language Version: Python 3.10+

## Setup Instructions
- Install dependencies:  
  `uv sync`
- Create and activate a virtual environment:  
  `uv venv && source .venv/bin/activate`
- Run python scripts with:  
  `uv run <path_to_script>.py`

## Package Management
- Add new packages:  
  `uv add <package>`
- Update all packages:  
  `uv pip upgrade`
- Check for outdated packages:  
  `uv pip list --outdated`
- Export requirements:  
  `uv pip freeze > requirements.txt`  
  or  
  `uv pip export > requirements.txt`  (from pyproject.toml if used)

## Testing Instructions
- TODO

## Code Style Guidelines
- Follow PEP8 conventions.
- Use type hints.
- Docstring format: Google
- Naming conventions: snake_case for functions/variables, PascalCase for classes.

## Commit & PR Guidance
- Commit messages: short, present tense.

## Security
- **Never commit `.env` files or credentials.**
- Use environment variables for secrets.

## Agent-Specific Notes
- Always refer to this AGENTS.md for workflow.
- Update documentation with feature changes.


