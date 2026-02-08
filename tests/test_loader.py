"""Tests for LanguageLoader."""

from unittest.mock import MagicMock, patch

import pytest

from yaduha.language import Language, LanguageNotFoundError, Sentence
from yaduha.loader import LanguageLoader


class SimpleSentenceForLoader(Sentence):
    """Simple test sentence type."""

    text: str

    def __str__(self) -> str:
        return self.text

    @classmethod
    def get_examples(cls) -> list[tuple[str, "SimpleSentenceForLoader"]]:
        return [("hello", cls(text="hi"))]


def create_test_language(code: str = "test") -> Language:
    """Helper to create a test language."""
    return Language(code=code, name="Test Language", sentence_types=(SimpleSentenceForLoader,))


def test_load_language_success() -> None:
    """Test loading a language via mocked entrypoints."""
    test_lang = create_test_language("ovp")

    mock_ep = MagicMock()
    mock_ep.load.return_value = test_lang

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep]
        mock_eps.return_value = mock_select

        lang = LanguageLoader.load_language("ovp")
        assert lang.code == "ovp"


def test_load_language_not_found() -> None:
    """Test that LanguageNotFoundError is raised for missing language."""
    mock_ep = MagicMock()
    mock_ep.load.return_value = create_test_language("other")

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep]
        mock_eps.return_value = mock_select

        with pytest.raises(LanguageNotFoundError):
            LanguageLoader.load_language("missing")


def test_load_language_entrypoint_error() -> None:
    """Test that LanguageNotFoundError is raised if entrypoint loading fails."""
    mock_ep = MagicMock()
    mock_ep.load.side_effect = ImportError("Package not found")

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep]
        mock_eps.return_value = mock_select

        with pytest.raises(LanguageNotFoundError):
            LanguageLoader.load_language("test")


def test_list_installed_languages() -> None:
    """Test listing all installed languages."""
    lang1 = create_test_language("test1")
    lang2 = create_test_language("test2")

    mock_ep1 = MagicMock()
    mock_ep1.load.return_value = lang1
    mock_ep2 = MagicMock()
    mock_ep2.load.return_value = lang2

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep1, mock_ep2]
        mock_eps.return_value = mock_select

        languages = LanguageLoader.list_installed_languages()
        assert len(languages) == 2
        assert lang1 in languages
        assert lang2 in languages


def test_list_installed_languages_skip_broken() -> None:
    """Test that broken entrypoints are skipped."""
    lang1 = create_test_language("test1")

    mock_ep1 = MagicMock()
    mock_ep1.load.return_value = lang1
    mock_ep2 = MagicMock()
    mock_ep2.load.side_effect = Exception("Error")

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep1, mock_ep2]
        mock_eps.return_value = mock_select

        languages = LanguageLoader.list_installed_languages()
        assert len(languages) == 1
        assert lang1 in languages


def test_list_installed_languages_empty() -> None:
    """Test listing languages when none are installed."""
    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = []
        mock_eps.return_value = mock_select

        languages = LanguageLoader.list_installed_languages()
        assert languages == []


def test_validate_language_success() -> None:
    """Test validating a properly implemented language."""
    test_lang = create_test_language("test")

    mock_ep = MagicMock()
    mock_ep.load.return_value = test_lang

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_entry_points = MagicMock()
        mock_entry_points.select.return_value = [mock_ep]
        mock_eps.return_value = mock_entry_points

        is_valid, errors = LanguageLoader.validate_language("test")
        assert is_valid
        assert errors == []


def test_validate_language_not_found() -> None:
    """Test validating a non-existent language."""
    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = []
        mock_eps.return_value = mock_select

        is_valid, errors = LanguageLoader.validate_language("missing")
        assert not is_valid
        assert len(errors) > 0




def test_validate_language_missing_str_method() -> None:
    """Test validating a language with sentence type missing __str__."""

    class BrokenSentence(Sentence):
        text: str

        def __str__(self) -> str:
            return self.text

        @classmethod
        def get_examples(cls) -> list[tuple[str, "BrokenSentence"]]:
            return []

    lang = Language(code="broken", name="Broken", sentence_types=(BrokenSentence,))

    # Modify to not have __str__
    delattr(BrokenSentence, "__str__")

    mock_ep = MagicMock()
    mock_ep.load.return_value = lang

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep]
        mock_eps.return_value = mock_select

        is_valid, errors = LanguageLoader.validate_language("broken")
        assert not is_valid
        assert any("__str__" in error for error in errors)


def test_validate_language_missing_get_examples() -> None:
    """Test validating a language with sentence type missing get_examples."""

    class NoExamplesSentence(Sentence):
        text: str

        def __str__(self) -> str:
            return self.text

        @classmethod
        def get_examples(cls) -> list[tuple[str, "NoExamplesSentence"]]:
            return []

    lang = Language(
        code="noexamples", name="No Examples", sentence_types=(NoExamplesSentence,)
    )

    # Remove get_examples
    delattr(NoExamplesSentence, "get_examples")

    mock_ep = MagicMock()
    mock_ep.load.return_value = lang

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep]
        mock_eps.return_value = mock_select

        is_valid, errors = LanguageLoader.validate_language("noexamples")
        assert not is_valid
        assert any("get_examples" in error for error in errors)


def test_validate_language_invalid_examples() -> None:
    """Test validating a language with invalid examples."""

    class InvalidExamplesSentence(Sentence):
        text: str

        def __str__(self) -> str:
            return self.text

        @classmethod
        def get_examples(cls) -> list[tuple[str, "InvalidExamplesSentence"]]:
            # Wrong return type
            return "not a list"  # type: ignore

    lang = Language(
        code="invalidex",
        name="Invalid Examples",
        sentence_types=(InvalidExamplesSentence,),
    )

    mock_ep = MagicMock()
    mock_ep.load.return_value = lang

    with patch("yaduha.loader.importlib.metadata.entry_points") as mock_eps:
        mock_select = MagicMock()
        mock_select.select.return_value = [mock_ep]
        mock_eps.return_value = mock_select

        is_valid, errors = LanguageLoader.validate_language("invalidex")
        assert not is_valid
        assert any("must return list" in error for error in errors)
