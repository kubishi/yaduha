from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, Type, TypeVar

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

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
    def get_examples(cls: Type[S]) -> List[Tuple[str, S]]:
        """Return example structured sentences and their English translations.

        Returns:
            List[Tuple[str, SentenceType]]: A list of tuples containing English
            translations and their corresponding structured sentences.
        """
        pass


@dataclass(frozen=True)
class VocabEntry:
    """Immutable vocabulary entry linking English and the target language."""

    english: str
    target: str


__all__ = [
    "Sentence",
    "VocabEntry",
    "Language",
    "LanguageNotFoundError",
    "LanguageValidationError",
]
