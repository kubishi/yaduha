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


# Import loader functions for external language loading
from yaduha.language.loader import (
    load_language_from_path,
    validate_language_path,
    LoadedLanguage,
    LanguageLoadError,
)

# Import git loader functions
from yaduha.language.git_loader import (
    load_language_from_git,
    test_language,
    get_repo_name,
    get_lang_path,
    remove_cached_language,
    list_cached_languages,
    GitLoadError,
    DEFAULT_CACHE_DIR,
)

__all__ = [
    "Sentence",
    "VocabEntry",
    # Path-based loading
    "load_language_from_path",
    "validate_language_path",
    "LoadedLanguage",
    "LanguageLoadError",
    # Git-based loading
    "load_language_from_git",
    "test_language",
    "get_repo_name",
    "get_lang_path",
    "remove_cached_language",
    "list_cached_languages",
    "GitLoadError",
    "DEFAULT_CACHE_DIR",
]
