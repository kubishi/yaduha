"""Tests for the evaluator module: Evaluator base class, OpenAIEvaluator, and batch_evaluate."""

from unittest.mock import MagicMock, patch

import pytest

from yaduha.evaluator import Evaluator, OpenAIEvaluator, batch_evaluate
from yaduha.translator import BackTranslation, Translation

# -- Evaluator base class --


class DummyEvaluator(Evaluator):
    name: str = "dummy"

    def evaluate(self, source: str, target: str) -> float:
        # Simple: fraction of words in common
        source_words = set(source.lower().split())
        target_words = set(target.lower().split())
        if not source_words:
            return 0.0
        return len(source_words & target_words) / len(source_words | target_words)


def test_evaluator_is_abstract() -> None:
    """Evaluator.evaluate raises NotImplementedError by default."""
    e = Evaluator(name="base")
    with pytest.raises(NotImplementedError):
        e.evaluate("a", "b")


def test_dummy_evaluator_identical() -> None:
    e = DummyEvaluator()
    assert e.evaluate("hello world", "hello world") == 1.0


def test_dummy_evaluator_different() -> None:
    e = DummyEvaluator()
    assert e.evaluate("hello world", "goodbye moon") == 0.0


def test_dummy_evaluator_partial() -> None:
    e = DummyEvaluator()
    score = e.evaluate("hello world", "hello moon")
    assert 0.0 < score < 1.0


# -- OpenAIEvaluator --


def _make_mock_embedding(vector: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.embedding = vector
    return mock


def test_openai_evaluator_identical_vectors() -> None:
    """Cosine similarity of identical vectors should be 1.0."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        vec = [1.0, 0.0, 0.0]
        mock_response = MagicMock()
        mock_response.data = [_make_mock_embedding(vec), _make_mock_embedding(vec)]
        mock_client.embeddings.create.return_value = mock_response

        evaluator = OpenAIEvaluator(api_key="test-key")
        score = evaluator.evaluate("hello", "hello")

        assert score == pytest.approx(1.0)
        mock_client.embeddings.create.assert_called_once()


def test_openai_evaluator_orthogonal_vectors() -> None:
    """Cosine similarity of orthogonal vectors should be 0.0."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            _make_mock_embedding([1.0, 0.0, 0.0]),
            _make_mock_embedding([0.0, 1.0, 0.0]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        evaluator = OpenAIEvaluator(api_key="test-key")
        score = evaluator.evaluate("hello", "world")

        assert score == pytest.approx(0.0)


def test_openai_evaluator_uses_correct_model() -> None:
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            _make_mock_embedding([1.0]),
            _make_mock_embedding([1.0]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        evaluator = OpenAIEvaluator(api_key="test-key", model="text-embedding-3-large")
        evaluator.evaluate("a", "b")

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input=["a", "b"]
        )


# -- batch_evaluate --


def _make_translation(source: str, back_source: str | None = None) -> Translation:
    bt = None
    if back_source is not None:
        bt = BackTranslation(source=back_source, target="target", translation_time=0.5)
    return Translation(
        source=source,
        target="target",
        translation_time=1.0,
        back_translation=bt,
    )


def test_batch_evaluate_with_back_translations() -> None:
    evaluator = DummyEvaluator()
    translations = [
        _make_translation("hello world", "hello world"),
        _make_translation("foo bar", "foo bar"),
    ]

    result = batch_evaluate(translations, evaluator)

    assert len(result) == 2
    assert result[0].evaluations["dummy"] == 1.0
    assert result[1].evaluations["dummy"] == 1.0


def test_batch_evaluate_skips_without_back_translation() -> None:
    evaluator = DummyEvaluator()
    translations = [
        _make_translation("hello world", "hello world"),
        _make_translation("no back translation"),
    ]

    result = batch_evaluate(translations, evaluator)

    assert len(result) == 2
    assert "dummy" in result[0].evaluations
    assert result[1].evaluations == {}


def test_batch_evaluate_preserves_existing_evaluations() -> None:
    evaluator = DummyEvaluator()
    t = _make_translation("hello", "hello")
    t = t.model_copy(update={"evaluations": {"existing": 0.5}})
    translations = [t]

    result = batch_evaluate(translations, evaluator)

    assert "existing" in result[0].evaluations
    assert "dummy" in result[0].evaluations
    assert result[0].evaluations["existing"] == 0.5


def test_batch_evaluate_does_not_mutate_originals() -> None:
    evaluator = DummyEvaluator()
    original = _make_translation("hello", "hello")
    translations = [original]

    result = batch_evaluate(translations, evaluator)

    assert original.evaluations == {}
    assert result[0].evaluations != {}


def test_batch_evaluate_empty_list() -> None:
    evaluator = DummyEvaluator()
    result = batch_evaluate([], evaluator)
    assert result == []
