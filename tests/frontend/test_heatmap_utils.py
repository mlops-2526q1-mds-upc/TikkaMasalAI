"""Unit tests for src/frontend/heatmap.py utilities."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
import pytest

from src.frontend.heatmap import (
    _try_parse_base64_image,
    find_heatmap_in_payload,
    overlay_heatmap_on_image,
)


def _png_base64_from_color(size=(8, 8), color=(0, 255, 0)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_try_parse_base64_image_valid_and_data_url():
    b64 = _png_base64_from_color()
    assert _try_parse_base64_image(b64) is not None

    data_url = f"data:image/png;base64,{b64}"
    assert _try_parse_base64_image(data_url) is not None


def test_try_parse_base64_image_invalid_returns_none():
    assert _try_parse_base64_image("not-base64!!") is None


def test_overlay_heatmap_on_image_smoke():
    base = Image.new("RGB", (32, 32), color=(255, 255, 255))
    hm = np.zeros((16, 16), dtype=float)
    hm[4:12, 4:12] = 10.0
    out = overlay_heatmap_on_image(base, hm, opacity=0.5)
    assert isinstance(out, Image.Image)
    assert out.size == base.size


def test_overlay_heatmap_3d_singleton_channel():
    base = Image.new("RGB", (10, 10))
    hm = np.ones((10, 10, 1), dtype=float)
    out = overlay_heatmap_on_image(base, hm)
    assert out.size == base.size


def test_overlay_heatmap_raises_on_invalid_shape():
    base = Image.new("RGB", (10, 10))
    hm = np.ones((2, 2, 3), dtype=float)  # not squeezable to 2D
    with pytest.raises(ValueError):
        overlay_heatmap_on_image(base, hm)


def test_find_heatmap_in_payload_overlay_base64():
    b64 = _png_base64_from_color()
    result = find_heatmap_in_payload({"overlay_base64": b64})
    assert "overlay_image" in result


def test_find_heatmap_in_payload_numeric_heatmap():
    payload = {"heatmap": [[0, 1], [2, 3]]}
    result = find_heatmap_in_payload(payload)
    assert "heatmap" in result
    hm = result["heatmap"]
    assert getattr(hm, "shape", None) == (2, 2)


def test_find_heatmap_in_payload_nested_image():
    b64 = _png_base64_from_color(size=(4, 4))
    payload = {"explanation": {"image_base64": b64}}
    result = find_heatmap_in_payload(payload)
    assert "overlay_image" in result
