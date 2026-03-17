"""Tests for yaduha.translator.agentic: AgenticTranslator."""

from tests.conftest import FakeAgent
from yaduha.translator.agentic import (
    AgenticTranslator,
    ConfidenceLevel,
    EvidenceItem,
    TranslationResponse,
)

# ---------------------------------------------------------------------------
# TranslationResponse model
# ---------------------------------------------------------------------------


def test_translation_response_construction():
    resp = TranslationResponse(
        translation="nüü üwi",
        confidence=ConfidenceLevel.HIGH,
        evidence=[
            EvidenceItem(tool_name="search", tool_input="sleep", tool_output="üwi"),
        ],
    )
    assert resp.translation == "nüü üwi"
    assert resp.confidence == ConfidenceLevel.HIGH
    assert len(resp.evidence) == 1


# ---------------------------------------------------------------------------
# AgenticTranslator.translate
# ---------------------------------------------------------------------------


def test_agentic_translate_basic():
    agent = FakeAgent()
    response = TranslationResponse(
        translation="nüü üwi",
        confidence=ConfidenceLevel.MEDIUM,
        evidence=[],
    )
    agent.set_response(response)

    translator = AgenticTranslator(agent=agent)
    result = translator.translate("I sleep.")

    assert result.source == "I sleep."
    assert result.target == "nüü üwi"
    assert result.translation_time > 0
    assert result.back_translation is None


def test_agentic_translate_with_evidence():
    agent = FakeAgent()
    evidence = [
        EvidenceItem(tool_name="search_english", tool_input="run", tool_output="poyoha"),
        EvidenceItem(tool_name="search_paiute", tool_input="nüü", tool_output="I, me"),
    ]
    response = TranslationResponse(
        translation="nüü poyoha",
        confidence=ConfidenceLevel.HIGH,
        evidence=evidence,
    )
    agent.set_response(response)

    translator = AgenticTranslator(agent=agent)
    result = translator.translate("I run.")

    assert result.target == "nüü poyoha"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 10


def test_agentic_translate_custom_system_prompt():
    agent = FakeAgent()
    response = TranslationResponse(
        translation="test",
        confidence=ConfidenceLevel.LOW,
        evidence=[],
    )
    agent.set_response(response)

    translator = AgenticTranslator(
        agent=agent,
        system_prompt="Custom prompt for testing.",
    )
    result = translator.translate("hello")
    assert result.target == "test"


def test_agentic_translate_via_call():
    """AgenticTranslator inherits Tool.__call__ which delegates to translate."""
    agent = FakeAgent()
    response = TranslationResponse(
        translation="target",
        confidence=ConfidenceLevel.HIGH,
        evidence=[],
    )
    agent.set_response(response)

    translator = AgenticTranslator(agent=agent)
    result = translator("test input")

    assert result.source == "test input"
    assert result.target == "target"
