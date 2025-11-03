from __future__ import annotations

import json

import httpx
import pytest


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.request = None  # not needed for success paths
        self._text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:  # pragma: no cover - unused in happy path
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        return self._payload

    @property
    def text(self):
        return self._text


class _DummyAsyncClient:
    def __init__(self, *_, **__):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url: str, json: dict):
        # mimic /api/generate success
        return _DummyResponse(
            200,
            {
                "response": "Hello from dummy model",
                "model": json.get("model", "gemma3:270m"),
                "done": True,
            },
        )

    async def get(self, url: str):
        # mimic /api/tags success
        return _DummyResponse(200, {"models": [{"name": "gemma3:270m"}]})


@pytest.fixture(autouse=True)
def _patch_httpx(monkeypatch):
    # Patch httpx.AsyncClient in the llm router so no real network calls occur
    from src.backend.routers import llm as llm_mod

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _DummyAsyncClient)
    yield


def test_llm_generate_success(client):
    payload = {"prompt": "Say hi", "temperature": 0.1, "max_tokens": 16}
    resp = client.post("/llm/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert "response" in data and isinstance(data["response"], str)


def test_llm_health_success(client):
    resp = client.get("/llm/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "gemma3:270m" in data.get("available_models", [])


def test_llm_health_connect_error(monkeypatch, client):
    from src.backend.routers import llm as llm_mod

    class _ConnectErrorClient:
        def __init__(self, *args, **kwargs):  # accept any args like httpx.AsyncClient
            pass

        async def __aenter__(self):  # pragma: no cover - trivial
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
            return False

        async def get(self, url: str):
            # Raise the same exception the router handles
            raise httpx.ConnectError("cannot connect", request=httpx.Request("GET", url))

    # Override the autouse dummy with our erroring client
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _ConnectErrorClient)

    resp = client.get("/llm/health")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Ollama service is not running. Please start Ollama first."


def test_llm_health_generic_error(monkeypatch, client):
    from src.backend.routers import llm as llm_mod

    class _GenericErrorClient:
        def __init__(self, *args, **kwargs):  # accept any args like httpx.AsyncClient
            pass

        async def __aenter__(self):  # pragma: no cover - trivial
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
            return False

        async def get(self, url: str):
            raise RuntimeError("boom")

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _GenericErrorClient)

    resp = client.get("/llm/health")
    assert resp.status_code == 500
    assert "Health check failed: boom" in resp.json()["detail"]


def test_llm_generate_404_model_not_found(monkeypatch, client):
    from src.backend.routers import llm as llm_mod

    class _AsyncClient404:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        class _Resp:
            status_code = 404

            def raise_for_status(self):  # pragma: no cover
                pass

            def json(self):  # pragma: no cover
                return {}

        async def post(self, url: str, json: dict):  # noqa: ARG002
            return self._Resp()

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _AsyncClient404)

    resp = client.post("/llm/generate", json={"prompt": "hi"})
    # NOTE: The router raises HTTPException inside try; the broad except Exception
    # catches it and wraps to 500. We assert current behavior and the message.
    assert resp.status_code == 500
    assert "Model 'gemma3:270m' not found" in resp.json().get("detail", "")


def test_llm_generate_connect_error(monkeypatch, client):
    from src.backend.routers import llm as llm_mod

    class _AsyncClientConnectErr:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):  # noqa: ARG002
            raise httpx.ConnectError("no conn", request=httpx.Request("POST", url))

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _AsyncClientConnectErr)

    resp = client.post("/llm/generate", json={"prompt": "hi"})
    assert resp.status_code == 503
    assert "Ollama service is not running" in resp.json().get("detail", "")


def test_llm_generate_timeout(monkeypatch, client):
    from src.backend.routers import llm as llm_mod

    class _AsyncClientTimeout:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):  # noqa: ARG002
            raise httpx.TimeoutException("too slow")

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _AsyncClientTimeout)

    resp = client.post("/llm/generate", json={"prompt": "hi"})
    assert resp.status_code == 504
    assert "timed out" in resp.json().get("detail", "").lower()


def test_llm_generate_httpstatuserror(monkeypatch, client):
    from src.backend.routers import llm as llm_mod

    class _AsyncClientHTTPError:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):  # noqa: ARG002
            req = httpx.Request("POST", url)
            resp = httpx.Response(500, request=req, text="boom")
            raise httpx.HTTPStatusError("server error", request=req, response=resp)

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _AsyncClientHTTPError)

    resp = client.post("/llm/generate", json={"prompt": "hi"})
    assert resp.status_code == 500
    assert resp.json().get("detail") == "Ollama API error: boom"


def test_llm_generate_generic_error(monkeypatch, client):
    from src.backend.routers import llm as llm_mod

    class _AsyncClientGeneric:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):  # noqa: ARG002
            raise ValueError("unexpected")

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _AsyncClientGeneric)

    resp = client.post("/llm/generate", json={"prompt": "hi"})
    assert resp.status_code == 500
    assert resp.json().get("detail") == "An unexpected error occurred: unexpected"
