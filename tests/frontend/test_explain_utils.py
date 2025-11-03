"""Unit tests for src/frontend/explain.py utilities."""

from __future__ import annotations

from src.frontend.explain import extract_top_scores


def test_extract_top_scores_from_predictions_mapping():
    payload = {"predictions": {"pizza": 0.9, "pasta": 0.1}}
    top = extract_top_scores(payload, top_n=2)
    assert top == [("pizza", 0.9), ("pasta", 0.1)]


def test_extract_top_scores_from_list_of_dicts():
    payload = {
        "predictions": [
            {"label": "ramen", "score": 0.7},
            {"label": "udon", "score": 0.3},
        ]
    }
    top = extract_top_scores(payload, top_n=1)
    assert top == [("ramen", 0.7)]


def test_extract_top_scores_falls_back_to_root_mapping():
    payload = {"soup": 0.2, "steak": 0.8}
    top = extract_top_scores(payload, top_n=2)
    assert top == [("steak", 0.8), ("soup", 0.2)]


def test_extract_top_scores_handles_empty_or_invalid():
    assert extract_top_scores({}, top_n=5) == []
    assert extract_top_scores([], top_n=5) == []  # type: ignore[arg-type]
