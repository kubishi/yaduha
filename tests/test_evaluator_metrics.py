"""Tests for chrF and BLEU evaluators (sacrebleu-based).

BERTScore and COMET are tested via mocking since they require heavy model downloads.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from yaduha.evaluator.bleu import BleuEvaluator
from yaduha.evaluator.chrf import ChrfEvaluator

# -- ChrfEvaluator --


class TestChrfEvaluator:
    def test_identical_strings_score_one(self) -> None:
        e = ChrfEvaluator()
        assert e.evaluate("The dog runs fast", "The dog runs fast") == pytest.approx(1.0)

    def test_empty_vs_nonempty_scores_zero(self) -> None:
        e = ChrfEvaluator()
        assert e.evaluate("hello world", "") == pytest.approx(0.0)

    def test_similar_strings_score_between_zero_and_one(self) -> None:
        e = ChrfEvaluator()
        score = e.evaluate("The dog runs fast", "The dog ran quickly")
        assert 0.0 < score < 1.0

    def test_completely_different_scores_low(self) -> None:
        e = ChrfEvaluator()
        score = e.evaluate("The dog runs fast", "A cat sleeps")
        assert score < 0.3

    def test_name_is_chrf(self) -> None:
        e = ChrfEvaluator()
        assert e.name == "chrf"

    def test_score_is_normalized(self) -> None:
        """Score should be between 0 and 1, not 0 and 100."""
        e = ChrfEvaluator()
        score = e.evaluate("hello", "hello")
        assert 0.0 <= score <= 1.0


# -- BleuEvaluator --


class TestBleuEvaluator:
    def test_identical_strings_score_one(self) -> None:
        e = BleuEvaluator()
        score = e.evaluate("The dog runs fast", "The dog runs fast")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_different_scores_zero(self) -> None:
        e = BleuEvaluator()
        score = e.evaluate("The dog runs fast", "A cat sleeps quietly")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        e = BleuEvaluator()
        score = e.evaluate("The dog runs fast", "The dog ran quickly")
        assert 0.0 < score < 1.0

    def test_name_is_bleu(self) -> None:
        e = BleuEvaluator()
        assert e.name == "bleu"

    def test_score_is_normalized(self) -> None:
        e = BleuEvaluator()
        score = e.evaluate("hello world", "hello world")
        assert 0.0 <= score <= 1.01  # sacrebleu can return slightly > 1.0 for sentence BLEU


# -- BertScoreEvaluator (mocked) --


class TestBertScoreEvaluator:
    def test_evaluate_calls_bert_score(self) -> None:
        mock_bert_score = MagicMock()
        mock_p = MagicMock()
        mock_p.item.return_value = 0.85
        mock_r = MagicMock()
        mock_r.item.return_value = 0.90
        mock_f1 = MagicMock()
        mock_f1.item.return_value = 0.87
        mock_bert_score.score.return_value = (mock_p, mock_r, mock_f1)

        with patch.dict(sys.modules, {"bert_score": mock_bert_score}):
            # Re-import to pick up the mocked module
            from yaduha.evaluator.bertscore import BertScoreEvaluator

            e = BertScoreEvaluator()
            result = e.evaluate("The dog runs", "The dog runs fast")

            assert result == pytest.approx(0.87)
            mock_bert_score.score.assert_called_once_with(
                ["The dog runs fast"],
                ["The dog runs"],
                model_type="microsoft/deberta-xlarge-mnli",
                lang="en",
                verbose=False,
            )

    def test_name_is_bertscore(self) -> None:
        from yaduha.evaluator.bertscore import BertScoreEvaluator

        e = BertScoreEvaluator()
        assert e.name == "bertscore"

    def test_custom_model_type(self) -> None:
        mock_bert_score = MagicMock()
        mock_t = MagicMock()
        mock_t.item.return_value = 0.9
        mock_bert_score.score.return_value = (mock_t, mock_t, mock_t)

        with patch.dict(sys.modules, {"bert_score": mock_bert_score}):
            from yaduha.evaluator.bertscore import BertScoreEvaluator

            e = BertScoreEvaluator(model_type="roberta-large")
            e.evaluate("a", "b")

            mock_bert_score.score.assert_called_once_with(
                ["b"], ["a"], model_type="roberta-large", lang="en", verbose=False
            )


# -- CometEvaluator (mocked) --


class TestCometEvaluator:
    def test_evaluate_calls_comet(self) -> None:
        mock_comet = MagicMock()
        mock_comet.download_model.return_value = "/tmp/model"
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.scores = [0.92]
        mock_model.predict.return_value = mock_output
        mock_comet.load_from_checkpoint.return_value = mock_model

        with patch.dict(sys.modules, {"comet": mock_comet}):
            from yaduha.evaluator.comet import CometEvaluator

            e = CometEvaluator()
            result = e.evaluate("The dog runs", "The dog runs fast")

            assert result == pytest.approx(0.92)
            mock_model.predict.assert_called_once_with(
                [{"src": "The dog runs", "mt": "The dog runs fast", "ref": "The dog runs"}],
                batch_size=1,
                gpus=0,
            )

    def test_name_is_comet(self) -> None:
        from yaduha.evaluator.comet import CometEvaluator

        e = CometEvaluator()
        assert e.name == "comet"

    def test_model_is_cached(self) -> None:
        mock_comet = MagicMock()
        mock_comet.download_model.return_value = "/tmp/model"
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.scores = [0.9]
        mock_model.predict.return_value = mock_output
        mock_comet.load_from_checkpoint.return_value = mock_model

        with patch.dict(sys.modules, {"comet": mock_comet}):
            from yaduha.evaluator.comet import CometEvaluator

            e = CometEvaluator()
            e.evaluate("a", "b")
            e.evaluate("c", "d")

            # Model should only be downloaded once
            mock_comet.download_model.assert_called_once()
            mock_comet.load_from_checkpoint.assert_called_once()
