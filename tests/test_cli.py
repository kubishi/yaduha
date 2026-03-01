"""Tests for Yaduha CLI."""

from unittest.mock import MagicMock, patch

from yaduha.cli import cmd_info, cmd_list, cmd_validate, main
from yaduha.language import Language, LanguageNotFoundError, Sentence


class SimpleCliSentence(Sentence):
    """Simple test sentence."""

    text: str

    def __str__(self) -> str:
        return self.text

    @classmethod
    def get_examples(cls) -> list[tuple[str, "SimpleCliSentence"]]:
        return [("hello", cls(text="hi"))]


def test_cmd_list_no_languages() -> None:
    """Test listing when no languages are installed."""
    with patch("yaduha.cli.LanguageLoader.list_installed_languages") as mock_list:
        mock_list.return_value = []

        result = cmd_list(MagicMock())
        assert result == 0


def test_cmd_list_with_languages() -> None:
    """Test listing with installed languages."""
    lang = Language(code="test", name="Test Language", sentence_types=(SimpleCliSentence,))

    with patch("yaduha.cli.LanguageLoader.list_installed_languages") as mock_list:
        mock_list.return_value = [lang]

        result = cmd_list(MagicMock())
        assert result == 0


def test_cmd_info_success() -> None:
    """Test getting language info."""
    lang = Language(code="ovp", name="Owens Valley Paiute", sentence_types=(SimpleCliSentence,))
    args = MagicMock()
    args.code = "ovp"

    with patch("yaduha.cli.LanguageLoader.load_language") as mock_load:
        mock_load.return_value = lang

        result = cmd_info(args)
        assert result == 0


def test_cmd_info_not_found() -> None:
    """Test info when language is not found."""
    args = MagicMock()
    args.code = "missing"

    with patch("yaduha.cli.LanguageLoader.load_language") as mock_load:
        mock_load.side_effect = LanguageNotFoundError("Not found")

        result = cmd_info(args)
        assert result == 1


def test_cmd_validate_success() -> None:
    """Test validating a valid language."""
    args = MagicMock()
    args.code = "ovp"

    with patch("yaduha.cli.LanguageLoader.validate_language") as mock_validate:
        mock_validate.return_value = (True, [])

        result = cmd_validate(args)
        assert result == 0


def test_cmd_validate_failure() -> None:
    """Test validating an invalid language."""
    args = MagicMock()
    args.code = "bad"

    with patch("yaduha.cli.LanguageLoader.validate_language") as mock_validate:
        mock_validate.return_value = (False, ["Missing __str__ method"])

        result = cmd_validate(args)
        assert result == 1


def test_main_languages_list() -> None:
    """Test main CLI with languages list command."""
    with patch("yaduha.cli.LanguageLoader.list_installed_languages") as mock_list:
        mock_list.return_value = []

        result = main(["languages", "list"])
        assert result == 0


def test_main_languages_info() -> None:
    """Test main CLI with languages info command."""
    lang = Language(code="test", name="Test", sentence_types=(SimpleCliSentence,))

    with patch("yaduha.cli.LanguageLoader.load_language") as mock_load:
        mock_load.return_value = lang

        result = main(["languages", "info", "test"])
        assert result == 0


def test_main_languages_validate() -> None:
    """Test main CLI with languages validate command."""
    with patch("yaduha.cli.LanguageLoader.validate_language") as mock_validate:
        mock_validate.return_value = (True, [])

        result = main(["languages", "validate", "test"])
        assert result == 0


def test_main_no_args() -> None:
    """Test main CLI with no arguments prints help."""
    result = main([])
    assert result == 0


def test_cmd_serve_starts_uvicorn() -> None:
    """Test that cmd_serve calls uvicorn.run."""
    args = MagicMock()
    args.host = "0.0.0.0"
    args.port = 8000

    mock_uvicorn = MagicMock()
    with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
        from yaduha.cli import cmd_serve

        cmd_serve(args)
        mock_uvicorn.run.assert_called_once()
