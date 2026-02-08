"""Load languages from installed language packages via entrypoints."""

import importlib.metadata
from typing import Any, List

from yaduha.language import Sentence
from yaduha.language.exceptions import LanguageNotFoundError, LanguageValidationError
from yaduha.language.language import Language


class LanguageLoader:
    """Load languages via entrypoints from installed language packages."""

    @staticmethod
    def load_language(language_code: str) -> Language:
        """Load a language by code via entrypoints.

        Looks for entry points in [project.entry-points."yaduha.languages"]

        Args:
            language_code: Language code to load (e.g., 'ovp')

        Returns:
            Language instance

        Raises:
            LanguageNotFoundError: If language is not found
        """
        try:
            entry_points = importlib.metadata.entry_points()
            # Handle both old and new entry_points() API
            if hasattr(entry_points, "select"):
                yaduha_languages = entry_points.select(group="yaduha.languages")  # type: ignore
            else:
                yaduha_languages = entry_points.get("yaduha.languages", [])  # type: ignore

            for ep in yaduha_languages:
                try:
                    language = ep.load()
                    if isinstance(language, Language) and language.code == language_code:
                        return language
                except Exception as e:
                    raise LanguageNotFoundError(
                        f"Failed to load entry point {ep.name}: {e}"
                    )

            raise LanguageNotFoundError(f"Language '{language_code}' not found")
        except LanguageNotFoundError:
            raise
        except Exception as e:
            raise LanguageNotFoundError(
                f"Failed to load language '{language_code}': {e}"
            )

    @staticmethod
    def list_installed_languages() -> List[Language]:
        """List all installed language packages via entrypoints.

        Returns:
            List of Language instances found
        """
        try:
            entry_points = importlib.metadata.entry_points()
            # Handle both old and new entry_points() API
            if hasattr(entry_points, "select"):
                yaduha_languages = entry_points.select(group="yaduha.languages")  # type: ignore
            else:
                yaduha_languages = entry_points.get("yaduha.languages", [])  # type: ignore

            languages: List[Language] = []
            for ep in yaduha_languages:
                try:
                    language = ep.load()
                    if isinstance(language, Language):
                        languages.append(language)
                except Exception:
                    continue

            return languages
        except Exception:
            return []

    @staticmethod
    def validate_language(language_code: str) -> tuple[bool, List[str]]:
        """Validate that a language package is properly implemented.

        Checks:
        - Language loads successfully
        - Has sentence types
        - Each type has __str__ method
        - Each type has get_examples() classmethod
        - Examples are valid instances
        - Sentence types are proper Pydantic models

        Args:
            language_code: Language code to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: List[str] = []

        try:
            language = LanguageLoader.load_language(language_code)
        except LanguageNotFoundError as e:
            return (False, [str(e)])

        if not language.sentence_types:
            return (False, ["Language has no sentence types"])

        for sentence_type in language.sentence_types:
            # Check 1: Has custom __str__ method
            try:
                if not hasattr(sentence_type, "__str__"):
                    errors.append(
                        f"{sentence_type.__name__} missing __str__ method"
                    )
                elif sentence_type.__str__ is Sentence.__str__:
                    errors.append(
                        f"{sentence_type.__name__} missing custom __str__ method"
                    )
            except Exception as e:
                errors.append(f"{sentence_type.__name__} __str__ check failed: {e}")

            # Check 2: Has get_examples classmethod
            if not hasattr(sentence_type, "get_examples"):
                errors.append(f"{sentence_type.__name__} missing get_examples()")
                continue

            # Check 3: Examples are valid
            try:
                examples = sentence_type.get_examples()
                if not isinstance(examples, list):
                    errors.append(
                        f"{sentence_type.__name__}.get_examples() must return list"
                    )
                    continue

                for i, item in enumerate(examples):
                    if not isinstance(item, tuple) or len(item) != 2:
                        errors.append(
                            f"{sentence_type.__name__} example {i} must be (str, instance) tuple"
                        )
                        continue

                    english, instance = item
                    if not isinstance(english, str):
                        errors.append(
                            f"{sentence_type.__name__} example {i} English must be str"
                        )
                    if not isinstance(instance, sentence_type):
                        errors.append(
                            f"{sentence_type.__name__} example {i} has wrong type"
                        )
                    else:
                        # Verify __str__ works
                        try:
                            str_result = str(instance)
                            if not str_result:
                                errors.append(
                                    f"{sentence_type.__name__} example {i} __str__ is empty"
                                )
                        except Exception as e:
                            errors.append(
                                f"{sentence_type.__name__} example {i} __str__ failed: {e}"
                            )
            except Exception as e:
                errors.append(
                    f"{sentence_type.__name__} get_examples() failed: {e}"
                )

            # Check 4: Is valid Pydantic model
            try:
                _ = sentence_type.model_fields
            except Exception as e:
                errors.append(
                    f"{sentence_type.__name__} is not valid Pydantic model: {e}"
                )

        return (len(errors) == 0, errors)
