"""Tests for Language class."""

import pytest

from yaduha.language import Language, Sentence


class SimpleSentence(Sentence):
    """Test sentence type."""

    text: str

    def __str__(self) -> str:
        return self.text

    @classmethod
    def get_examples(cls) -> list[tuple[str, "SimpleSentence"]]:
        return [("hello", cls(text="nüü"))]


class AnotherSentence(Sentence):
    """Another test sentence type."""

    word: str

    def __str__(self) -> str:
        return self.word

    @classmethod
    def get_examples(cls) -> list[tuple[str, "AnotherSentence"]]:
        return []


def test_language_creation() -> None:
    """Test creating a Language instance."""
    lang = Language(
        code="test",
        name="Test Language",
        sentence_types=(SimpleSentence, AnotherSentence),
    )
    assert lang.code == "test"
    assert lang.name == "Test Language"
    assert len(lang.sentence_types) == 2


def test_language_invalid_code() -> None:
    """Test that invalid code raises ValueError."""
    with pytest.raises(ValueError, match="code must be"):
        Language(code="", name="Test", sentence_types=(SimpleSentence,))

    with pytest.raises(ValueError, match="code must be"):
        Language(code=None, name="Test", sentence_types=(SimpleSentence,))  # type: ignore


def test_language_invalid_name() -> None:
    """Test that invalid name raises ValueError."""
    with pytest.raises(ValueError, match="name must be"):
        Language(code="test", name="", sentence_types=(SimpleSentence,))

    with pytest.raises(ValueError, match="name must be"):
        Language(code="test", name=None, sentence_types=(SimpleSentence,))  # type: ignore


def test_language_empty_sentence_types() -> None:
    """Test that empty sentence_types raises ValueError."""
    with pytest.raises(ValueError, match="sentence_types must not be empty"):
        Language(code="test", name="Test", sentence_types=())


def test_language_invalid_sentence_type() -> None:
    """Test that non-Sentence type raises TypeError."""

    class NotASentence:
        pass

    with pytest.raises(TypeError, match="not a Sentence subclass"):
        Language(code="test", name="Test", sentence_types=(NotASentence,))  # type: ignore


def test_language_repr() -> None:
    """Test Language __repr__."""
    lang = Language(
        code="ovp",
        name="Owens Valley Paiute",
        sentence_types=(SimpleSentence, AnotherSentence),
    )
    assert "ovp" in repr(lang)
    assert "2" in repr(lang)


def test_language_equality() -> None:
    """Test Language equality based on code."""
    lang1 = Language(code="test", name="Test", sentence_types=(SimpleSentence,))
    lang2 = Language(code="test", name="Different", sentence_types=(AnotherSentence,))
    lang3 = Language(code="other", name="Test", sentence_types=(SimpleSentence,))

    assert lang1 == lang2  # Same code
    assert lang1 != lang3  # Different code
    assert lang1 != "test"  # Not a Language


def test_language_hashable() -> None:
    """Test that Language instances are hashable."""
    lang1 = Language(code="test", name="Test", sentence_types=(SimpleSentence,))
    lang2 = Language(code="test", name="Test", sentence_types=(SimpleSentence,))

    # Should be usable in sets/dicts
    lang_set = {lang1, lang2}
    assert len(lang_set) == 1

    lang_dict = {lang1: "value"}
    assert lang_dict[lang2] == "value"
