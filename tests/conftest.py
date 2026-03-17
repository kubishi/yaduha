"""Shared test fixtures for yaduha tests."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field

from yaduha.agent import Agent, AgentResponse
from yaduha.language import Language, Sentence

# ---------------------------------------------------------------------------
# FakeAgent — deterministic Agent for testing translators/tools without mocks
# ---------------------------------------------------------------------------


class FakeAgent(Agent):
    """Agent that returns pre-configured responses without any network calls."""

    model: str = "fake-model"
    name: ClassVar[str] = "fake_agent"

    # The content to return (str or BaseModel instance)
    _response_content: str | BaseModel = ""
    _prompt_tokens: int = 5
    _completion_tokens: int = 10

    def set_response(
        self, content: str | BaseModel, *, prompt_tokens: int = 5, completion_tokens: int = 10
    ):
        """Configure the next response this agent will return."""
        self._response_content = content
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    def get_response(self, messages, response_format=str, tools=None):
        content = self._response_content
        if response_format is not str and isinstance(content, str):
            # If caller expects a BaseModel but we have a string, try parsing it
            content = response_format.model_validate_json(content)
        return AgentResponse(
            content=content,
            response_time=0.01,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )


# ---------------------------------------------------------------------------
# SimpleSentence — minimal sentence type for testing
# ---------------------------------------------------------------------------


class SimpleSentence(Sentence):
    """Minimal sentence type: subject + verb."""

    subject: str = Field(description="The subject")
    verb: str = Field(description="The verb")

    def __str__(self) -> str:
        return f"{self.subject} {self.verb}"

    @classmethod
    def get_examples(cls) -> list[tuple[str, SimpleSentence]]:
        return [
            ("I sleep.", cls(subject="nüü", verb="üwi")),
            ("You run.", cls(subject="üü", verb="poyoha")),
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_agent() -> FakeAgent:
    """FakeAgent with a default text response."""
    agent = FakeAgent()
    agent.set_response("translated text")
    return agent


@pytest.fixture
def simple_language() -> Language:
    """Minimal Language with one sentence type."""
    return Language(code="test", name="Test Language", sentence_types=(SimpleSentence,))


@pytest.fixture
def tmp_jsonl(tmp_path: pathlib.Path) -> pathlib.Path:
    """Temporary .jsonl file path for JsonLogger tests."""
    return tmp_path / "test_log.jsonl"
