"""Tests for InstructionsBackTranslator."""

from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from yaduha.agent import Agent, AgentResponse
from yaduha.translator import BackTranslation
from yaduha.translator.instructions_back import (
    BackTranslationResponse,
    InstructionsBackTranslator,
    _build_source_code_instructions,
    _get_package_sources,
)

# -- Helpers --


class FakeAgent(Agent[str]):
    """Concrete Agent subclass for testing."""

    name: ClassVar[str] = "fake_agent"
    model: str = "fake-model"
    _response: AgentResponse | None = None
    _call_args: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def set_response(self, response: AgentResponse) -> None:
        self._response = response

    def get_response(self, messages, response_format=str, tools=None) -> AgentResponse:
        self._call_args = {
            "messages": messages,
            "response_format": response_format,
            "tools": tools,
        }
        if self._response is not None:
            return self._response
        return AgentResponse(content="default", response_time=0.1, prompt_tokens=0, completion_tokens=0)


def _make_back_translation_response(
    english_translation: str | None = "The dog runs",
    grammatical: bool = True,
    reasoning: str = "Analyzed morphemes.",
    prompt_tokens: int = 20,
    completion_tokens: int = 10,
) -> AgentResponse:
    response_content = BackTranslationResponse(
        reasoning=reasoning,
        grammatical=grammatical,
        english_translation=english_translation,
    )
    return AgentResponse(
        content=response_content,
        response_time=0.5,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# -- BackTranslationResponse --


class TestBackTranslationResponse:
    def test_grammatical_with_translation(self) -> None:
        r = BackTranslationResponse(
            reasoning="Traced through code.",
            grammatical=True,
            english_translation="The dog runs",
        )
        assert r.grammatical is True
        assert r.english_translation == "The dog runs"

    def test_ungrammatical_with_null_translation(self) -> None:
        r = BackTranslationResponse(
            reasoning="Cannot parse.",
            grammatical=False,
            english_translation=None,
        )
        assert r.grammatical is False
        assert r.english_translation is None

    def test_reasoning_is_required(self) -> None:
        with pytest.raises(Exception):
            BackTranslationResponse(
                grammatical=True,
                english_translation="test",
            )

    def test_grammatical_is_required(self) -> None:
        with pytest.raises(Exception):
            BackTranslationResponse(
                reasoning="some reasoning",
                english_translation="test",
            )


# -- InstructionsBackTranslator.back_translate --


class TestBackTranslate:
    def test_returns_back_translation(self) -> None:
        agent = FakeAgent()
        agent.set_response(_make_back_translation_response(english_translation="The dog runs"))
        bt = InstructionsBackTranslator(agent=agent, instructions="Analyze OVP")

        result = bt.back_translate("nüü mu-puni-ku")

        assert isinstance(result, BackTranslation)
        assert result.source == "The dog runs"
        assert result.target == "nüü mu-puni-ku"

    def test_passes_instructions_as_system_message(self) -> None:
        agent = FakeAgent()
        agent.set_response(_make_back_translation_response())
        bt = InstructionsBackTranslator(agent=agent, instructions="Source code instructions")

        bt.back_translate("test-input")

        assert agent._call_args is not None
        messages = agent._call_args["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Source code instructions"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test-input"

    def test_requests_structured_response_format(self) -> None:
        agent = FakeAgent()
        agent.set_response(_make_back_translation_response())
        bt = InstructionsBackTranslator(agent=agent, instructions="Analyze")

        bt.back_translate("test")

        assert agent._call_args is not None
        assert agent._call_args["response_format"] is BackTranslationResponse

    def test_token_counts_forwarded(self) -> None:
        agent = FakeAgent()
        agent.set_response(_make_back_translation_response(prompt_tokens=100, completion_tokens=50))
        bt = InstructionsBackTranslator(agent=agent, instructions="Analyze")

        result = bt.back_translate("test")

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    def test_translation_time_is_positive(self) -> None:
        agent = FakeAgent()
        agent.set_response(_make_back_translation_response())
        bt = InstructionsBackTranslator(agent=agent, instructions="Analyze")

        result = bt.back_translate("test")

        assert result.translation_time > 0

    def test_null_translation_becomes_empty_string(self) -> None:
        """When english_translation is None, source should be empty string."""
        agent = FakeAgent()
        agent.set_response(
            _make_back_translation_response(
                english_translation=None,
                grammatical=False,
                reasoning="Unintelligible.",
            )
        )
        bt = InstructionsBackTranslator(agent=agent, instructions="Analyze")

        result = bt.back_translate("gibberish")

        assert result.source == ""


# -- InstructionsBackTranslator.from_language --


class TestFromLanguage:
    def test_creates_instance_from_language(self) -> None:
        agent = FakeAgent()
        mock_language = MagicMock()
        mock_language.sentence_types = []

        with (
            patch(
                "yaduha.translator.instructions_back.LanguageLoader"
            ) as mock_loader,
            patch(
                "yaduha.translator.instructions_back._build_source_code_instructions"
            ) as mock_build,
        ):
            mock_loader.load_language.return_value = mock_language
            mock_build.return_value = "Generated instructions"

            bt = InstructionsBackTranslator.from_language("ovp", agent=agent)

            mock_loader.load_language.assert_called_once_with("ovp")
            mock_build.assert_called_once_with(mock_language, 10)
            assert bt.instructions == "Generated instructions"
            assert bt.agent is agent

    def test_custom_n_examples(self) -> None:
        agent = FakeAgent()
        mock_language = MagicMock()
        mock_language.sentence_types = []

        with (
            patch(
                "yaduha.translator.instructions_back.LanguageLoader"
            ) as mock_loader,
            patch(
                "yaduha.translator.instructions_back._build_source_code_instructions"
            ) as mock_build,
        ):
            mock_loader.load_language.return_value = mock_language
            mock_build.return_value = "instructions"

            InstructionsBackTranslator.from_language("ovp", agent=agent, n_examples=25)

            mock_build.assert_called_once_with(mock_language, 25)


# -- _get_package_sources --


class TestGetPackageSources:
    def test_raises_for_no_module(self) -> None:
        mock_language = MagicMock()
        mock_sentence_type = MagicMock()
        mock_sentence_type.__name__ = "FakeSentence"
        mock_language.sentence_types = [mock_sentence_type]

        with patch("inspect.getmodule", return_value=None):
            with pytest.raises(ValueError, match="Cannot find module"):
                _get_package_sources(mock_language)


# -- _build_source_code_instructions --


class TestBuildSourceCodeInstructions:
    def test_includes_preamble(self) -> None:
        mock_language = MagicMock()
        mock_language.sentence_types = []

        with patch(
            "yaduha.translator.instructions_back._get_package_sources",
            return_value={"mymod": "# source code"},
        ):
            result = _build_source_code_instructions(mock_language, n_examples=0)

        assert "Python source code" in result
        assert "## Source Code: mymod" in result
        assert "# source code" in result

    def test_includes_examples_section(self) -> None:
        mock_language = MagicMock()
        mock_sentence_type = MagicMock(spec=[])  # empty spec so hasattr(sample_iter) is False
        mock_sentence_type.__name__ = "SVSentence"
        mock_sentence = MagicMock()
        mock_sentence.__str__ = lambda self: "nüü puni-ku"
        mock_sentence.model_dump_json.return_value = '{"subject": "nüü"}'
        mock_sentence_type.get_examples = MagicMock(return_value=[("input", mock_sentence)])
        mock_language.sentence_types = [mock_sentence_type]

        with patch(
            "yaduha.translator.instructions_back._get_package_sources",
            return_value={},
        ):
            result = _build_source_code_instructions(mock_language, n_examples=5)

        assert "## Examples" in result
        assert "SVSentence" in result
