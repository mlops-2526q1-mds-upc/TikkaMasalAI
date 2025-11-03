from __future__ import annotations


def test_explain_endpoint_rejects_non_image_and_logs_guard(monkeypatch, client):
    from src.backend.routers import predict as predict_mod

    class BoomLogger:
        def warning(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("forced logger failure")

    monkeypatch.setattr(predict_mod, "logger", BoomLogger())

    files = {"image": ("note.txt", b"not-image", "text/plain")}
    resp = client.post("/predict/explain", files=files)
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()


def test_explain_endpoint_raises_500_on_exception(monkeypatch, client, tiny_png_bytes):
    from src.backend.routers import predict as predict_mod

    def explode(_img):  # noqa: ANN001
        raise RuntimeError("kaboom")

    monkeypatch.setattr(predict_mod, "generate_attention_heatmap", explode)

    files = {"image": ("tiny.png", tiny_png_bytes, "image/png")}
    resp = client.post("/predict/explain", files=files)
    assert resp.status_code == 500
    assert "Explainability generation failed: kaboom" in resp.json().get("detail", "")
