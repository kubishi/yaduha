from pydantic import BaseModel
from dataclasses import dataclass
from typing import Generic, List, Tuple, TypeVar, Type
from abc import ABC, abstractmethod

S = TypeVar("S", bound="Sentence")

class Sentence(BaseModel, ABC, Generic[S]):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_examples(cls: Type[S]) -> List[Tuple[str, S]]:
        """Return example structured sentences and their English translations.

        Returns:
            List[Tuple[str, SentenceType]]: A list of tuples containing English translations
            and their corresponding structured sentences.
        """
        pass

@dataclass(frozen=True)
class VocabEntry:
    """Immutable vocabulary entry linking English and the target language"""
    english: str
    target: str
