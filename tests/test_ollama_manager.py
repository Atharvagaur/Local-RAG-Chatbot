"""Tests for ollama_manager.py.

Uses a mocked requests session so no Ollama server is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import requests

import pytest

from backend.ollama_manager import OllamaError, OllamaManager, OllamaModel


class _FakeResponse:
    def __init__(self, json_data, status_code=200, raise_error=False):
        self._json = json_data
        self.status_code = status_code
        self.raise_error = raise_error

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.raise_error:
            raise requests.HTTPError("boom")


def _patch_get(monkeypatch, payload, status_code=200, raise_error=False):
    def fake_get(self, url, timeout=None):
        return _FakeResponse(payload, status_code, raise_error)

    monkeypatch.setattr(requests.Session, "get", fake_get)


def test_model_catalog_splits_chat_and_embedding(monkeypatch):
    payload = {
        "models": [
            {
                "name": "qwen3:4b",
                "digest": "abc",
                "size": 100,
                "modified_at": "t",
            },
            {
                "name": "mxbai-embed-large:latest",
                "digest": "def",
                "size": 200,
                "modified_at": "t",
            },
            {
                "name": "gemma4:31b-cloud",
                "digest": "ghi",
                "size": 300,
                "modified_at": "t",
            },
        ]
    }
    _patch_get(monkeypatch, payload)
    om = OllamaManager("http://localhost:11434")
    catalog = om.get_catalog()

    chat_names = [m.name for m in catalog.chat_models]
    assert "qwen3:4b" in chat_names
    assert "gemma4:31b-cloud" not in chat_names  # cloud excluded
    assert "mxbai-embed-large:latest" in chat_names  # non-cloud present

    embed_names = [m.name for m in catalog.embedding_models]
    assert "mxbai-embed-large:latest" in embed_names


def test_model_exists_and_validate(monkeypatch):
    payload = {"models": [{"name": "qwen3:4b", "digest": "abc", "size": 1, "modified_at": "t"}]}
    _patch_get(monkeypatch, payload)
    om = OllamaManager("http://localhost:11434")
    assert om.model_exists("qwen3:4b") is True
    assert om.model_exists("missing:model") is False

    om.validate_model("qwen3:4b")  # should not raise
    with pytest.raises(OllamaError):
        om.validate_model("missing:model")


def test_is_available_false_when_unreachable(monkeypatch):
    def fake_get(self, url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(requests.Session, "get", fake_get)
    om = OllamaManager("http://localhost:11434")
    assert om.is_available() is False
