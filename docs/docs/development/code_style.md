# Code Style Guide

This is the single source of truth for code style and engineering conventions in TikkaMasalAI. Other docs (e.g. `CONTRIBUTING.md`, `AGENTS.md`) should point here instead of duplicating rules.

## 1. Philosophy
Consistency, clarity, and maintainability first; performance second (unless a hotspot). Prefer explicitness over cleverness. Lean on tooling (Ruff) to automate style so humans focus on logic.

## 2. Tooling Overview
- **Formatter & Linter:** Ruff (formatter + lint + import sorting). No Black/isort/flake8.
- **Type checking:** (Planned) — use type hints now; future stricter enforcement may add mypy/pyright.
- **Tests:** pytest.
- **Packaging & deps:** `uv` (`pyproject.toml` + `uv.lock`).

Authoritative configuration lives in `pyproject.toml` — do not restate values here (e.g., line length = 99) except for explanation.

## 3. Formatting Rules (Enforced by Ruff)
- Line length: 99 characters (soft limit). Wrap proactively for readability.
- Imports auto-sorted by Ruff rule `I`.
- No manual alignment with spaces.
- Use trailing commas in multi-line literals where Ruff applies them.
- One public class/function per core module concept when possible; split large modules.

## 4. Naming Conventions
| Element              | Style        | Example                     |
|----------------------|--------------|-----------------------------|
| Packages / Modules   | snake_case   | `food_loader.py`            |
| Functions / Methods  | snake_case   | `load_dataset()`            |
| Variables            | snake_case   | `train_samples`             |
| Constants            | UPPER_SNAKE  | `DEFAULT_BATCH_SIZE`        |
| Classes              | PascalCase   | `FoodClassifier`            |
| Test Functions       | snake_case   | `test_resnet_inference()`   |
| Private helpers      | _snake_case  | `_select_device()`          |

Avoid ambiguous abbreviations (prefer `num_classes` over `n_cls`).

## 5. Docstrings (Google Style)
Use Google style docstrings for public modules, classes, and non-trivial functions.

Minimal function example:
```python
def classify(image: bytes) -> int:
    """Return predicted class index for an input image.

    Args:
        image: Encoded image bytes (RGB or convertible).
    Returns:
        Integer class index (0-based) present in `src/labels.py`.
    Raises:
        ValueError: If decoding fails.
    """
    ...
```

Class example:
```python
class FoodClassifier:
    """Abstract interface for all food classification models."""
```

Omit docstrings ONLY for truly obvious one-liners or overridden methods whose behavior is unchanged — otherwise keep them.

## 6. Type Hints
- All new functions must have return type hints.
- Use `collections.abc` types (`Mapping`, `Sequence`) instead of concrete containers when appropriate.
- Avoid `Any`; justify with a short comment if unavoidable.
- Prefer `Path` from `pathlib` for filesystem paths.
- Use `|` (PEP 604) union syntax (`str | None`).

## 7. Logging vs Printing
- Use `logging` for library / model code (`logging.getLogger(__name__)`).
- `print()` acceptable only in CLI entry scripts, notebooks, or quick debug code not committed.

## 8. Testing Conventions
- Test files live in `tests/` and begin with `test_`.
- Function naming: `test_<unit_of_behavior>()`.
- Follow Arrange–Act–Assert sections (comment optional for clarity).
- Keep tests deterministic: set seeds (e.g., `torch.manual_seed(42)`).
- Avoid network/dataset downloads inside tests; mock or sample locally cached artifacts.
- Add a regression test for each bug fix.

## 9. Imports
Order (enforced automatically):
1. Standard library
2. Third-party
3. First-party (`src.*`)
4. Local relative

Guidelines:
- No wildcard imports (`from x import *`).
- Explicit re-exports: define `__all__` only when curating API surfaces.
- Prefer absolute imports from `src.` over deep relative imports.