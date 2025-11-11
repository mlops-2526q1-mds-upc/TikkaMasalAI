from __future__ import annotations

import io

from PIL import Image
import pytest
import torch


def test_generate_attention_heatmap_raises_when_no_attentions(monkeypatch, tiny_png_bytes):
    from src.backend.routers import predict as predict_mod

    img = Image.open(io.BytesIO(tiny_png_bytes)).convert("RGB")

    class Out:
        def __init__(self):
            self.logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
            self.attentions = None  # trigger ValueError branch

    class FakeModel:
        def __call__(self, **kwargs):  # noqa: ANN003
            return Out()

        def eval(self):  # pragma: no cover - setup
            return self

        def to(self, device):  # pragma: no cover - setup
            return self

        def set_attn_implementation(self, mode: str):  # pragma: no cover - optional
            return None

    class FakeProcessor:
        def __call__(self, images, return_tensors="pt"):  # noqa: ARG002
            return {"pixel_values": torch.zeros(1, 3, 16, 16)}

    monkeypatch.setattr(
        predict_mod,
        "_load_model_and_preprocessor",
        lambda: (FakeModel(), FakeProcessor()),
    )

    with pytest.raises(ValueError) as e:
        predict_mod.generate_attention_heatmap(img)
    assert "Model did not return attention weights" in str(e.value)


def test_generate_attention_heatmap_success_minimal(monkeypatch, tiny_png_bytes):
    from src.backend.routers import predict as predict_mod

    img = Image.open(io.BytesIO(tiny_png_bytes)).convert("RGB")

    # Minimal, consistent outputs: tokens=5 -> n_patches = tokens-1 = 4 => 2x2 grid
    tokens = 5
    heads = 2

    class Out:
        def __init__(self):
            # logits for 3 classes
            self.logits = torch.tensor([[0.1, 0.6, 0.3]], dtype=torch.float32)
            # attentions: [batch=1, heads=2, tokens=5, tokens=5]
            att = torch.ones(1, heads, tokens, tokens, dtype=torch.float32)
            self.attentions = [att]

    class FakeModel:
        def __call__(self, **kwargs):  # noqa: ANN003
            # ensure output_attentions=True is accepted
            return Out()

        def eval(self):  # pragma: no cover
            return self

        def to(self, device):  # pragma: no cover
            return self

        def set_attn_implementation(self, mode: str):  # pragma: no cover
            return None

    class FakeProcessor:
        def __call__(self, images, return_tensors="pt"):  # noqa: ARG002
            return {"pixel_values": torch.zeros(1, 3, 16, 16)}

    monkeypatch.setattr(
        predict_mod,
        "_load_model_and_preprocessor",
        lambda: (FakeModel(), FakeProcessor()),
    )

    result = predict_mod.generate_attention_heatmap(img)

    # Validate returned structure
    assert set(result.keys()) >= {
        "predicted_class",
        "confidence",
        "attention_map",
        "num_layers",
        "num_heads",
        "grid_size",
    }
    # confidence is numeric
    assert isinstance(result["confidence"], float)
    # attention_map is base64-like non-empty string
    assert isinstance(result["attention_map"], str) and len(result["attention_map"]) > 0
    # we set one layer and 2 heads
    assert result["num_layers"] == 1
    assert result["num_heads"] == heads
    # for tokens=5, we expect 2x2 grid after processing
    assert result["grid_size"] == "2x2"


def test_generate_attention_heatmap_attn_impl_raises(monkeypatch, tiny_png_bytes):
    from src.backend.routers import predict as predict_mod

    img = Image.open(io.BytesIO(tiny_png_bytes)).convert("RGB")

    tokens = 5
    heads = 1

    class Out:
        def __init__(self):
            self.logits = torch.tensor([[0.2, 0.7, 0.1]], dtype=torch.float32)
            self.attentions = [torch.ones(1, heads, tokens, tokens, dtype=torch.float32)]

    class FakeModel:
        def set_attn_implementation(self, mode: str):
            raise RuntimeError("not supported")

        def __call__(self, **kwargs):  # noqa: ANN003
            return Out()

        def eval(self):  # pragma: no cover
            return self

        def to(self, device):  # pragma: no cover
            return self

    class FakeProcessor:
        def __call__(self, images, return_tensors="pt"):  # noqa: ARG002
            return {"pixel_values": torch.zeros(1, 3, 16, 16)}

    monkeypatch.setattr(
        predict_mod,
        "_load_model_and_preprocessor",
        lambda: (FakeModel(), FakeProcessor()),
    )

    # Should not raise despite set_attn_implementation throwing
    out = predict_mod.generate_attention_heatmap(img)
    assert "attention_map" in out and isinstance(out["attention_map"], str)
