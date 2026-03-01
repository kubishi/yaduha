"""Tests for yaduha.api: routes and dependencies."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from yaduha.api.app import create_app
from yaduha.api.dependencies import (
    _resolve_api_key,
    create_agent,
    get_language,
    get_sentence_type,
)
from yaduha.api.models import AgentConfig
from yaduha.language import Language

from tests.conftest import SimpleSentence


# ---------------------------------------------------------------------------
# Dependencies unit tests
# ---------------------------------------------------------------------------


class TestResolveApiKey:
    def test_provider_specific_header(self):
        key = _resolve_api_key("openai", {"x-openai-key": "sk-test"})
        assert key == "sk-test"

    def test_generic_header(self):
        key = _resolve_api_key("openai", {"x-api-key": "sk-generic"})
        assert key == "sk-generic"

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        key = _resolve_api_key("openai", {})
        assert key == "sk-env"

    def test_provider_header_takes_priority(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        key = _resolve_api_key("openai", {"x-openai-key": "sk-header", "x-api-key": "sk-generic"})
        assert key == "sk-header"

    def test_ollama_returns_none(self):
        key = _resolve_api_key("ollama", {})
        assert key is None

    def test_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        key = _resolve_api_key("openai", {})
        assert key is None


class TestCreateAgent:
    def test_unknown_provider_raises_400(self):
        config = AgentConfig(provider="unknown", model="test")
        with pytest.raises(HTTPException) as exc_info:
            create_agent(config, {})
        assert exc_info.value.status_code == 400

    def test_missing_key_raises_401(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = AgentConfig(provider="openai", model="gpt-4o")
        with pytest.raises(HTTPException) as exc_info:
            create_agent(config, {})
        assert exc_info.value.status_code == 401

    def test_valid_openai_config(self):
        config = AgentConfig(provider="openai", model="gpt-4o")
        agent = create_agent(config, {"x-openai-key": "sk-test"})
        from yaduha.agent.openai import OpenAIAgent
        assert isinstance(agent, OpenAIAgent)

    def test_ollama_no_key_needed(self):
        config = AgentConfig(provider="ollama", model="llama3")
        agent = create_agent(config, {})
        from yaduha.agent.ollama import OllamaAgent
        assert isinstance(agent, OllamaAgent)


class TestGetLanguage:
    def test_found(self):
        lang = Language(code="test", name="Test", sentence_types=(SimpleSentence,))
        with patch("yaduha.api.dependencies.LanguageLoader.load_language", return_value=lang):
            result = get_language("test")
            assert result.code == "test"

    def test_not_found_raises_404(self):
        from yaduha.language import LanguageNotFoundError
        with patch(
            "yaduha.api.dependencies.LanguageLoader.load_language",
            side_effect=LanguageNotFoundError("test"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                get_language("test")
            assert exc_info.value.status_code == 404


class TestGetSentenceType:
    def test_found(self):
        lang = Language(code="test", name="Test", sentence_types=(SimpleSentence,))
        result = get_sentence_type(lang, "SimpleSentence")
        assert result is SimpleSentence

    def test_not_found_raises_404(self):
        lang = Language(code="test", name="Test", sentence_types=(SimpleSentence,))
        with pytest.raises(HTTPException) as exc_info:
            get_sentence_type(lang, "NonExistent")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# API route tests via TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    with patch("yaduha.api.routes.health.LanguageLoader.list_installed_languages", return_value=[]):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["languages_available"] == 0


def test_health_with_languages(client):
    lang = Language(code="test", name="Test", sentence_types=(SimpleSentence,))
    with patch("yaduha.api.routes.health.LanguageLoader.list_installed_languages", return_value=[lang]):
        response = client.get("/api/health")
        assert response.json()["languages_available"] == 1


def test_list_languages(client):
    lang = Language(code="test", name="Test Language", sentence_types=(SimpleSentence,))
    with patch("yaduha.api.routes.languages.LanguageLoader.list_installed_languages", return_value=[lang]):
        response = client.get("/api/languages")
        assert response.status_code == 200
        data = response.json()
        assert len(data["languages"]) == 1
        assert data["languages"][0]["code"] == "test"
        assert data["languages"][0]["name"] == "Test Language"
        assert data["languages"][0]["sentence_type_count"] == 1


def test_list_languages_empty(client):
    with patch("yaduha.api.routes.languages.LanguageLoader.list_installed_languages", return_value=[]):
        response = client.get("/api/languages")
        assert response.status_code == 200
        assert response.json()["languages"] == []


def test_get_language_info(client):
    lang = Language(code="test", name="Test", sentence_types=(SimpleSentence,))
    with patch("yaduha.api.dependencies.LanguageLoader.load_language", return_value=lang):
        response = client.get("/api/languages/test")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "test"
        assert len(data["sentence_types"]) == 1
        assert data["sentence_types"][0]["name"] == "SimpleSentence"


def test_get_language_info_not_found(client):
    from yaduha.language import LanguageNotFoundError

    with patch(
        "yaduha.api.dependencies.LanguageLoader.load_language",
        side_effect=LanguageNotFoundError("missing"),
    ):
        response = client.get("/api/languages/missing")
        assert response.status_code == 404
