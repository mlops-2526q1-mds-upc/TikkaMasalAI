from __future__ import annotations

from typing import Any, Dict


def test_predict_rejects_non_image(client):
    files = {"image": ("not_image.txt", b"hello", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()


def test_predict_success_with_mock(client, monkeypatch, tiny_png_bytes):
    # Arrange: patch classify_food to avoid loading real model
    from src.backend.routers import predict as predict_mod

    def fake_classify_food(_img) -> Dict[str, float]:
        return {"pizza": 0.91, "hamburger": 0.05, "sushi": 0.04}

    monkeypatch.setattr(predict_mod, "classify_food", fake_classify_food)

    files = {"image": ("tiny.png", tiny_png_bytes, "image/png")}

    # Act
    resp = client.post("/predict", files=files)

    # Assert
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "tiny.png"
    assert "predictions" in data
    assert list(data["predictions"])  # non-empty


def test_explain_success_with_mock(client, monkeypatch, tiny_png_bytes):
    # Arrange: patch generate_attention_heatmap to return a small, deterministic payload
    from src.backend.routers import predict as predict_mod

    def fake_heatmap(_img) -> Dict[str, Any]:
        return {
            "predicted_class": "pizza",
            "confidence": 0.99,
            "attention_map": "iVBORfakeBase64==",
            "num_layers": 12,
            "num_heads": 12,
            "grid_size": "14x14",
        }

    monkeypatch.setattr(predict_mod, "generate_attention_heatmap", fake_heatmap)

    files = {"image": ("tiny.png", tiny_png_bytes, "image/png")}

    # Act
    resp = client.post("/predict/explain", files=files)

    # Assert
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "tiny.png"
    assert data["predicted_class"] == "pizza"
    assert isinstance(data["confidence"], float)
