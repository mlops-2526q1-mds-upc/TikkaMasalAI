"""Unit tests for src/frontend/predict.py utility functions."""

from __future__ import annotations

from src.frontend.predict import extract_primary_label


def test_extract_primary_label_simple_label():
    payload = {"label": "pizza"}
    assert extract_primary_label(payload) == "pizza"


def test_extract_primary_label_nested_label():
    payload = {"prediction": {"label": "sushi"}}
    assert extract_primary_label(payload) == "sushi"


def test_extract_primary_label_scores_dict():
    payload = {"predictions": {"burger": 0.2, "pasta": 0.8}}
    assert extract_primary_label(payload) == "pasta"


def test_extract_primary_label_non_string_value():
    payload = {"class": 5}
    assert extract_primary_label(payload) == "5"


def test_extract_primary_label_none_when_absent():
    assert extract_primary_label({}) is None
