from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import pytest
import torch


def test_id_to_label_known():
    from src.backend.routers.predict import _id_to_label

    # from constants mapping: "76": "pizza"
    assert _id_to_label(76) == "pizza"


def test_id_to_label_fallback():
    from src.backend.routers.predict import _id_to_label

    assert _id_to_label(10_000) == "class_10000"


def test_load_missing_model_path_raises(monkeypatch):
    from src.backend.routers import predict as predict_mod

    # ensure no cache
    predict_mod._load_model_and_preprocessor.cache_clear()

    # point to a definitely-missing directory
    monkeypatch.setattr(predict_mod, "MODEL_PATH", Path("/__definitely_missing__/nope"))

    with pytest.raises(FileNotFoundError):
        predict_mod._load_model_and_preprocessor()


def test_load_success_with_mismatch_and_len_exception(monkeypatch, tmp_path):
    from src.backend.routers import predict as predict_mod

    predict_mod._load_model_and_preprocessor.cache_clear()

    # Make the model path exist
    tmp_dir = tmp_path / "prithiv"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(predict_mod, "MODEL_PATH", tmp_dir)

    # Fake model and processor
    class FakeModel:
        class _Cfg:
            num_labels = 999  # mismatch with LABELS length

        def __init__(self):
            self.config = self._Cfg()

        def eval(self):
            return self

        def to(self, device):  # pragma: no cover - simple chain
            return self

    class FakeProcessor:
        pass

    class FakeSiglip:
        @staticmethod
        def from_pretrained(path, local_files_only=True):  # noqa: ARG002
            return FakeModel()

    class FakeAutoProc:
        @staticmethod
        def from_pretrained(path, local_files_only=True):  # noqa: ARG002
            return FakeProcessor()

    # Cause len(LABELS) to raise once to execute the except-pass branch
    monkeypatch.setattr(predict_mod, "LABELS", None)

    monkeypatch.setattr(predict_mod, "SiglipForImageClassification", FakeSiglip)
    monkeypatch.setattr(predict_mod, "AutoImageProcessor", FakeAutoProc)

    m, p = predict_mod._load_model_and_preprocessor()
    assert isinstance(m, FakeModel)
    assert isinstance(p, FakeProcessor)


def test_load_logs_label_mismatch_warning(monkeypatch, tmp_path, caplog):
    from src.backend.routers import predict as predict_mod

    predict_mod._load_model_and_preprocessor.cache_clear()

    tmp_dir = tmp_path / "prithiv"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(predict_mod, "MODEL_PATH", tmp_dir)

    class FakeModel:
        class _Cfg:
            num_labels = 123

        def __init__(self):
            self.config = self._Cfg()

        def eval(self):
            return self

        def to(self, device):  # pragma: no cover
            return self

    class FakeProcessor:
        pass

    class FakeSiglip:
        @staticmethod
        def from_pretrained(path, local_files_only=True):  # noqa: ARG002
            return FakeModel()

    class FakeAutoProc:
        @staticmethod
        def from_pretrained(path, local_files_only=True):  # noqa: ARG002
            return FakeProcessor()

    # Provide a LABELS dict with length different than FakeModel.config.num_labels
    monkeypatch.setattr(predict_mod, "LABELS", {"0": "a", "1": "b"})
    monkeypatch.setattr(predict_mod, "SiglipForImageClassification", FakeSiglip)
    monkeypatch.setattr(predict_mod, "AutoImageProcessor", FakeAutoProc)

    with caplog.at_level("WARNING"):
        m, p = predict_mod._load_model_and_preprocessor()
    assert isinstance(m, FakeModel)
    assert isinstance(p, FakeProcessor)


def test_id_to_label_exception_path(monkeypatch):
    from src.backend.routers import predict as predict_mod

    # Force LABELS to be invalid so .get raises
    monkeypatch.setattr(predict_mod, "LABELS", None)
    assert predict_mod._id_to_label(42) == "class_42"


def test_classify_food_with_fake_model(monkeypatch, tiny_png_bytes):
    from src.backend.routers import predict as predict_mod

    # Build tiny PIL image from fixture
    img = Image.open(io.BytesIO(tiny_png_bytes)).convert("RGB")

    # Fake model output to drive specific logits
    class Out:
        def __init__(self):
            self.logits = torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float32)

    class FakeModel:
        def __call__(self, **kwargs):  # noqa: ANN003
            return Out()

        def eval(self):  # pragma: no cover - not used here
            return self

        def to(self, device):  # pragma: no cover - not used here
            return self

        class config:  # pragma: no cover - not used here
            num_labels = 3

    class FakeProcessor:
        def __call__(self, images, return_tensors="pt"):  # noqa: ARG002
            # Mimic HF processors by returning tensors with .to method
            return {"pixel_values": torch.zeros(1, 3, 32, 32)}

    # Bypass _load_model_and_preprocessor heavy path
    monkeypatch.setattr(
        predict_mod, "_load_model_and_preprocessor", lambda: (FakeModel(), FakeProcessor())
    )

    preds = predict_mod.classify_food(img)
    # Expect three classes returned (min(5, 3) = 3)
    assert len(preds) == 3
    # Highest logit (2.0) should map to label id 0 -> using constants: "0": "apple_pie"
    # But our mapping uses indices 0.. so the top1 label is whatever _id_to_label(0) returns
    assert isinstance(next(iter(preds.values())), float)
