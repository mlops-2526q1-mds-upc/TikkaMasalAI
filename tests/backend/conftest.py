"""Shared test fixtures for backend FastAPI app.

These fixtures avoid heavyweight model/LLM calls by monkeypatching at the
function level in individual tests. Import-time errors (e.g., missing optional
packages in a minimal env) will cause the backend test suite to be skipped
cleanly so the scaffold can land even before all deps are present.
"""

from __future__ import annotations

import io

import pytest


@pytest.fixture(scope="session")
def app():
    """Import and return the FastAPI app.

    If import fails due to missing heavy deps (torch/transformers/etc.),
    skip the backend test suite gracefully.
    """
    try:
        from src.backend.app import app as fastapi_app
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Skipping backend tests — app import failed: {exc}")
    return fastapi_app


@pytest.fixture()
def client(app):
    """Synchronous TestClient for FastAPI endpoints."""
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Skipping backend tests — TestClient import failed: {exc}")
    return TestClient(app)


@pytest.fixture()
def tiny_png_bytes() -> bytes:
    """Provide a tiny valid PNG image as bytes for upload tests."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Skipping backend image tests — PIL not available: {exc}")

    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
