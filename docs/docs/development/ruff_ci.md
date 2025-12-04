# Ruff in CI

This repository runs Ruff in GitHub Actions to enforce linting and import sorting.

### CI Workflow

- Defined in `.github/workflows/ruff.yml`.
- Uses Python 3.10, installs `ruff`, runs `ruff check .`.
- Pip cache improves install speed.

### Local usage

```bash
ruff check .          # lint
ruff check --fix .    # auto-fix
ruff format           # apply Ruff formatter
```

For style rules and broader engineering conventions, see Development → [Code Style Guide](./code_style.md).

### Pre-commit (Ruff only)

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

___

- Configuration lives in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`.
- Tests are included in linting; some docstring rules are ignored for `tests/**`.
