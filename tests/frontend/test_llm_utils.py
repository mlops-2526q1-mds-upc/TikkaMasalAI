"""Unit tests for src/frontend/llm.py utilities."""

from __future__ import annotations

from src.frontend.llm import extract_llm_text


def test_extract_llm_text_simple_keys():
    assert extract_llm_text({"text": "hello"}) == "hello"
    assert extract_llm_text({"response": "world"}) == "world"


def test_extract_llm_text_openai_chat_like():
    payload = {"choices": [{"message": {"content": "hi there"}}]}
    assert extract_llm_text(payload) == "hi there"


def test_extract_llm_text_openai_completion_like():
    payload = {"choices": [{"text": "completion"}]}
    assert extract_llm_text(payload) == "completion"


def test_extract_llm_text_nested_data():
    payload = {"data": {"text": "nested"}}
    assert extract_llm_text(payload) == "nested"


def test_extract_llm_text_returns_none_when_absent():
    assert extract_llm_text({}) is None
