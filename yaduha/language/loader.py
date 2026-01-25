"""
Dynamic Language Loader

Loads user-created language modules from external directories or repositories.
This allows users to define their own languages and use them with
the yaduha translation pipeline.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass

from yaduha.language import Sentence, VocabEntry


class LanguageLoadError(Exception):
    """Raised when a language module fails to load or validate."""
    pass


@dataclass
class LoadedLanguage:
    """Container for a dynamically loaded language module."""

    name: str
    code: str
    description: str
    sentence_types: Tuple[Type[Sentence], ...]
    module: Any

    # Vocabulary (optional, may not be exposed)
    nouns: List[VocabEntry]
    transitive_verbs: List[VocabEntry]
    intransitive_verbs: List[VocabEntry]
    adjectives: List[VocabEntry]
    adverbs: List[VocabEntry]

    def get_all_examples(self) -> List[Tuple[str, Sentence]]:
        """Get examples from all sentence types."""
        examples = []
        for sentence_type in self.sentence_types:
            examples.extend(sentence_type.get_examples())
        return examples


def load_language_from_path(path: Union[str, Path], module_name: Optional[str] = None) -> LoadedLanguage:
    """
    Load a language module from a directory path.

    The directory should contain:
    - __init__.py or language.py with the language definition
    - vocab.py with vocabulary definitions

    Required exports from the language module:
    - LANGUAGE_NAME: str - Name of the language
    - LANGUAGE_CODE: str - Short code for the language
    - SENTENCE_TYPES: Tuple[Type[Sentence], ...] - Available sentence types

    Optional exports:
    - LANGUAGE_DESCRIPTION: str - Description of the language
    - NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS, ADJECTIVES, ADVERBS: Lists of VocabEntry

    Args:
        path: Path to the language directory
        module_name: Optional name for the module (defaults to directory name)

    Returns:
        LoadedLanguage: Container with the loaded language

    Raises:
        LanguageLoadError: If the language fails to load or validate
    """
    path = Path(path).resolve()

    if not path.is_dir():
        raise LanguageLoadError(f"Path is not a directory: {path}")

    # Determine module name
    if module_name is None:
        module_name = f"yaduha_lang_{path.name}"

    # Find the main module file
    init_file = path / "__init__.py"
    language_file = path / "language.py"

    if init_file.exists():
        main_file = init_file
    elif language_file.exists():
        main_file = language_file
    else:
        raise LanguageLoadError(
            f"Language directory must contain __init__.py or language.py: {path}"
        )

    # Load the module
    try:
        # Add parent directory to path temporarily for relative imports
        parent_dir = str(path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        spec = importlib.util.spec_from_file_location(module_name, main_file)
        if spec is None or spec.loader is None:
            raise LanguageLoadError(f"Failed to create module spec for: {main_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    except Exception as e:
        raise LanguageLoadError(f"Failed to load module from {main_file}: {e}") from e

    # Validate required exports
    return _validate_and_wrap_module(module, path)


def _validate_and_wrap_module(module: Any, path: Path) -> LoadedLanguage:
    """Validate a loaded module and wrap it in a LoadedLanguage container."""

    errors = []

    # Check required exports
    if not hasattr(module, 'LANGUAGE_NAME'):
        errors.append("Missing required export: LANGUAGE_NAME")
    if not hasattr(module, 'LANGUAGE_CODE'):
        errors.append("Missing required export: LANGUAGE_CODE")
    if not hasattr(module, 'SENTENCE_TYPES'):
        errors.append("Missing required export: SENTENCE_TYPES")

    if errors:
        raise LanguageLoadError(
            f"Language at {path} is missing required exports:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    # Validate SENTENCE_TYPES
    sentence_types = module.SENTENCE_TYPES
    if not isinstance(sentence_types, (list, tuple)):
        raise LanguageLoadError(
            f"SENTENCE_TYPES must be a list or tuple, got {type(sentence_types)}"
        )

    for i, st in enumerate(sentence_types):
        if not isinstance(st, type):
            raise LanguageLoadError(
                f"SENTENCE_TYPES[{i}] must be a class, got {type(st)}"
            )
        if not issubclass(st, Sentence):
            raise LanguageLoadError(
                f"SENTENCE_TYPES[{i}] ({st.__name__}) must be a subclass of Sentence"
            )
        # Check for get_examples method
        if not hasattr(st, 'get_examples'):
            raise LanguageLoadError(
                f"Sentence type {st.__name__} must implement get_examples() classmethod"
            )
        # Check for __str__ method
        if not hasattr(st, '__str__') or st.__str__ is object.__str__:
            raise LanguageLoadError(
                f"Sentence type {st.__name__} must implement __str__() method"
            )

    # Validate that examples work
    for st in sentence_types:
        try:
            examples = st.get_examples()
            if not examples:
                raise LanguageLoadError(
                    f"Sentence type {st.__name__}.get_examples() returned empty list"
                )
            for english, sentence in examples:
                if not isinstance(english, str):
                    raise LanguageLoadError(
                        f"Example english must be str, got {type(english)}"
                    )
                if not isinstance(sentence, Sentence):
                    raise LanguageLoadError(
                        f"Example sentence must be Sentence instance, got {type(sentence)}"
                    )
                # Try rendering the sentence
                rendered = str(sentence)
                if not isinstance(rendered, str) or not rendered.strip():
                    raise LanguageLoadError(
                        f"Sentence.__str__() must return non-empty string, got: {repr(rendered)}"
                    )
        except LanguageLoadError:
            raise
        except Exception as e:
            raise LanguageLoadError(
                f"Failed to validate examples for {st.__name__}: {e}"
            ) from e

    # Get optional exports with defaults
    description = getattr(module, 'LANGUAGE_DESCRIPTION', '')
    nouns = getattr(module, 'NOUNS', [])
    transitive_verbs = getattr(module, 'TRANSITIVE_VERBS', [])
    intransitive_verbs = getattr(module, 'INTRANSITIVE_VERBS', [])
    adjectives = getattr(module, 'ADJECTIVES', [])
    adverbs = getattr(module, 'ADVERBS', [])

    return LoadedLanguage(
        name=module.LANGUAGE_NAME,
        code=module.LANGUAGE_CODE,
        description=description,
        sentence_types=tuple(sentence_types),
        module=module,
        nouns=nouns,
        transitive_verbs=transitive_verbs,
        intransitive_verbs=intransitive_verbs,
        adjectives=adjectives,
        adverbs=adverbs,
    )


def validate_language_path(path: Union[str, Path]) -> List[str]:
    """
    Validate a language directory without fully loading it.

    Returns a list of validation errors (empty list if valid).
    """
    errors = []
    path = Path(path).resolve()

    if not path.exists():
        errors.append(f"Path does not exist: {path}")
        return errors

    if not path.is_dir():
        errors.append(f"Path is not a directory: {path}")
        return errors

    # Check for required files
    init_file = path / "__init__.py"
    language_file = path / "language.py"
    vocab_file = path / "vocab.py"

    if not init_file.exists() and not language_file.exists():
        errors.append("Missing __init__.py or language.py")

    if not vocab_file.exists():
        errors.append("Missing vocab.py (vocabulary file)")

    return errors
