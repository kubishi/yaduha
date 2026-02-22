"""Tests for InstructionsTranslator with back_translator and evaluators."""

from typing import ClassVar
from unittest.mock import MagicMock

from yaduha.agent import Agent, AgentResponse
from yaduha.evaluator import Evaluator
from yaduha.translator import BackTranslation, Translation
from yaduha.translator.back_translator import BackTranslator
from yaduha.translator.instructions import InstructionsTranslator

# -- Helpers --


class FakeAgent(Agent[str]):
    """Concrete Agent subclass for testing."""

    name: ClassVar[str] = "fake_agent"
    model: str = "fake-model"
    _response: AgentResponse | None = None

    model_config = {"arbitrary_types_allowed": True}

    def set_response(self, content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self._response = AgentResponse(
            content=content,
            response_time=0.5,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def get_response(self, messages, response_format=str, tools=None) -> AgentResponse:
        if self._response is not None:
            return self._response
        return AgentResponse(content="default", response_time=0.1, prompt_tokens=0, completion_tokens=0)


class FakeBackTranslator(BackTranslator):
    """Concrete BackTranslator for testing."""

    name: ClassVar[str] = "fake_back_translator"
    description: ClassVar[str] = "Fake back translator for testing."
    back_source: str = "back translated text"

    def back_translate(self, text: str) -> BackTranslation:
        return BackTranslation(
            source=self.back_source,
            target=text,
            translation_time=0.3,
            prompt_tokens=5,
            completion_tokens=3,
        )


class StubEvaluator(Evaluator):
    name: str = "stub"

    def evaluate(self, source: str, target: str) -> float:
        return 0.75


class StubEvaluator2(Evaluator):
    name: str = "stub2"

    def evaluate(self, source: str, target: str) -> float:
        return 0.5


# -- Construction --


class TestInstructionsTranslatorConstruction:
    def test_minimal_construction(self) -> None:
        agent = FakeAgent()
        t = InstructionsTranslator(agent=agent, instructions="Translate to OVP")
        assert t.back_translator is None
        assert list(t.evaluators) == []

    def test_with_back_translator(self) -> None:
        agent = FakeAgent()
        bt = FakeBackTranslator()
        t = InstructionsTranslator(agent=agent, instructions="Translate", back_translator=bt)
        assert t.back_translator is not None

    def test_with_evaluators(self) -> None:
        agent = FakeAgent()
        t = InstructionsTranslator(
            agent=agent,
            instructions="Translate",
            evaluators=[StubEvaluator(), StubEvaluator2()],
        )
        assert len(list(t.evaluators)) == 2


# -- translate() --


class TestInstructionsTranslatorTranslate:
    def test_basic_translation(self) -> None:
        agent = FakeAgent()
        agent.set_response("translated-text")
        t = InstructionsTranslator(agent=agent, instructions="Translate to OVP")

        result = t.translate("The dog runs")

        assert isinstance(result, Translation)
        assert result.source == "The dog runs"
        assert result.target == "translated-text"
        assert result.back_translation is None
        assert result.evaluations == {}

    def test_token_counts_forwarded(self) -> None:
        agent = FakeAgent()
        agent.set_response("text", prompt_tokens=42, completion_tokens=17)
        t = InstructionsTranslator(agent=agent, instructions="Translate")

        result = t.translate("test")

        assert result.prompt_tokens == 42
        assert result.completion_tokens == 17

    def test_translation_time_is_positive(self) -> None:
        agent = FakeAgent()
        agent.set_response("text")
        t = InstructionsTranslator(agent=agent, instructions="Translate")

        result = t.translate("test")

        assert result.translation_time > 0

    def test_with_back_translator(self) -> None:
        agent = FakeAgent()
        agent.set_response("ovp-text")
        bt = FakeBackTranslator(back_source="The dog runs fast")
        t = InstructionsTranslator(
            agent=agent,
            instructions="Translate",
            back_translator=bt,
        )

        result = t.translate("The dog runs fast")

        assert result.back_translation is not None
        assert result.back_translation.source == "The dog runs fast"
        assert result.back_translation.target == "ovp-text"

    def test_with_back_translator_and_evaluators(self) -> None:
        agent = FakeAgent()
        agent.set_response("ovp-text")
        bt = FakeBackTranslator(back_source="The dog runs")
        t = InstructionsTranslator(
            agent=agent,
            instructions="Translate",
            back_translator=bt,
            evaluators=[StubEvaluator(), StubEvaluator2()],
        )

        result = t.translate("The dog runs")

        assert result.back_translation is not None
        assert "stub" in result.evaluations
        assert "stub2" in result.evaluations
        assert result.evaluations["stub"] == 0.75
        assert result.evaluations["stub2"] == 0.5

    def test_evaluators_without_back_translator_produces_no_scores(self) -> None:
        """Evaluators require back-translation to run."""
        agent = FakeAgent()
        agent.set_response("text")
        t = InstructionsTranslator(
            agent=agent,
            instructions="Translate",
            evaluators=[StubEvaluator()],
        )

        result = t.translate("test")

        assert result.back_translation is None
        assert result.evaluations == {}

    def test_evaluator_receives_source_and_back_translated_source(self) -> None:
        """Evaluator.evaluate should be called with (original_source, back_translation.source)."""
        agent = FakeAgent()
        agent.set_response("ovp-text")
        bt = FakeBackTranslator(back_source="Back translated text")
        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.name = "mock_eval"
        mock_evaluator.evaluate.return_value = 0.9

        t = InstructionsTranslator(
            agent=agent,
            instructions="Translate",
            back_translator=bt,
            evaluators=[mock_evaluator],
        )

        result = t.translate("Original text")

        mock_evaluator.evaluate.assert_called_once_with("Original text", "Back translated text")
        assert result.evaluations["mock_eval"] == 0.9
