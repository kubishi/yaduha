"""Tests for yaduha.translator.pipeline: PipelineTranslator.translate()."""

from unittest.mock import patch

from tests.conftest import FakeAgent, SimpleSentence
from yaduha.agent import AgentResponse
from yaduha.language import Language
from yaduha.translator.pipeline import PipelineTranslator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_for_pipeline():
    """Create a FakeAgent that can handle both structured and text responses."""
    agent = FakeAgent()
    return agent


# ---------------------------------------------------------------------------
# from_language
# ---------------------------------------------------------------------------


def test_from_language_loads():
    mock_language = Language(code="test", name="Test Language", sentence_types=(SimpleSentence,))
    with patch(
        "yaduha.translator.pipeline.LanguageLoader.load_language", return_value=mock_language
    ):
        agent = FakeAgent()
        translator = PipelineTranslator.from_language("test", agent)
        assert translator.SentenceType == (SimpleSentence,)


def test_from_language_not_found():
    from yaduha.language import LanguageNotFoundError

    with patch(
        "yaduha.translator.pipeline.LanguageLoader.load_language",
        side_effect=LanguageNotFoundError("test"),
    ):
        import pytest

        agent = FakeAgent()
        with pytest.raises(LanguageNotFoundError):
            PipelineTranslator.from_language("test", agent)


# ---------------------------------------------------------------------------
# translate() with mocked agent
# ---------------------------------------------------------------------------


def test_translate_returns_translation():
    """Test full pipeline: English → structured → render → back-translate."""
    from yaduha.tool.english_to_sentences import SentenceList

    agent = FakeAgent()

    # First call: EnglishToSentences — return a SentenceList
    sentence = SimpleSentence(subject="nüü", verb="üwi")
    sentence_list = SentenceList[SimpleSentence](sentences=[sentence])
    english_to_sentences_response = AgentResponse(
        content=sentence_list,
        response_time=0.1,
        prompt_tokens=10,
        completion_tokens=5,
    )

    # Second call: SentenceToEnglish — return English text
    sentence_to_english_response = AgentResponse(
        content="I sleep.",
        response_time=0.05,
        prompt_tokens=5,
        completion_tokens=3,
    )

    with patch.object(
        FakeAgent,
        "get_response",
        side_effect=[english_to_sentences_response, sentence_to_english_response],
    ):
        translator = PipelineTranslator(
            agent=agent,
            SentenceType=(SimpleSentence,),
        )
        result = translator.translate("I sleep.")

    assert result.source == "I sleep."
    assert result.target == "Nüü üwi."  # SimpleSentence.__str__, cleaned
    assert result.back_translation is not None
    assert result.back_translation.source == "I sleep."
    assert result.translation_time > 0
