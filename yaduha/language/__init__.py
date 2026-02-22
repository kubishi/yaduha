from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from yaduha.language.exceptions import LanguageNotFoundError, LanguageValidationError
from yaduha.language.language import Language

S = TypeVar("S", bound="Sentence")


class Sentence(BaseModel, ABC, Generic[S]):
    """Base class for all sentence types in Yaduha.

    All language packages must define sentence types that inherit from this class.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Render this sentence in the target language."""
        pass

    @classmethod
    @abstractmethod
    def get_examples(cls: type[S]) -> list[tuple[str, S]]:
        """Return example structured sentences and their English translations.

        Returns:
            List[Tuple[str, SentenceType]]: A list of tuples containing English
            translations and their corresponding structured sentences.
        """
        pass


class VocabEntry(BaseModel):
    """Vocabulary entry linking English and the target language."""

    model_config = ConfigDict(frozen=True)

    english: str
    target: str


__all__ = [
    "Sentence",
    "VocabEntry",
    "Language",
    "LanguageNotFoundError",
    "LanguageValidationError",
]
