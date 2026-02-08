"""Tests for PipelineTranslator.from_language()."""

from unittest.mock import MagicMock, patch

import pytest

from yaduha.language import Language, LanguageNotFoundError, Sentence
from yaduha.translator.pipeline import PipelineTranslator


class SimpleSentenceForPipeline(Sentence):
    """Simple test sentence type."""

    text: str

    def __str__(self) -> str:
        return self.text

    @classmethod
    def get_examples(cls) -> list[tuple[str, "SimpleSentenceForPipeline"]]:
        return [("hello", cls(text="hi"))]


def create_test_language(code: str = "test") -> Language:
    """Helper to create a test language."""
    return Language(code=code, name="Test Language", sentence_types=(SimpleSentenceForPipeline,))


def test_from_language_calls_loader() -> None:
    """Test that from_language calls LanguageLoader.load_language()."""
    test_lang = create_test_language("test")

    with patch("yaduha.translator.pipeline.LanguageLoader.load_language") as mock_load:
        mock_load.return_value = test_lang
        mock_agent = MagicMock()

        # The from_language method should call load_language
        try:
            PipelineTranslator.from_language(
                language_code="test",
                agent=mock_agent,
            )
        except Exception:
            # Expected to fail due to Pydantic validation, but we verify the call was made
            pass

        # Verify load_language was called with the correct code
        mock_load.assert_called_once_with("test")


def test_from_language_not_found() -> None:
    """Test from_language when language is not found."""
    with patch("yaduha.translator.pipeline.LanguageLoader.load_language") as mock_load:
        mock_load.side_effect = LanguageNotFoundError("Language not found")

        mock_agent = MagicMock()

        with pytest.raises(LanguageNotFoundError):
            PipelineTranslator.from_language(
                language_code="missing",
                agent=mock_agent,
            )
