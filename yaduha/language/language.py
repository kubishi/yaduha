"""Language class for wrapping sentence types and metadata."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class Language:
    """Container for a language's sentence types and metadata.

    Attributes:
        code: ISO 639-3 language code (e.g., 'ovp')
        name: Human-readable language name (e.g., 'Owens Valley Paiute')
        sentence_types: Tuple of Sentence subclasses supported by this language
    """

    def __init__(
        self,
        code: str,
        name: str,
        sentence_types: tuple[type[Any], ...],
        get_instructions: Callable[[], str] | None = None,
    ) -> None:
        """Initialize a Language instance.

        Args:
            code: Language code identifier
            name: Human-readable language name
            sentence_types: Tuple of Sentence subclasses
            get_instructions: Optional callable that returns natural language
                grammar instructions (vocabulary, rules, examples) suitable
                for use as an LLM system prompt.

        Raises:
            ValueError: If code or name is empty, or sentence_types is empty
            TypeError: If sentence_types contains non-Sentence types
        """
        # Import here to avoid circular imports
        from yaduha.language import Sentence

        if not code or not isinstance(code, str):
            raise ValueError("code must be a non-empty string")
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        if not sentence_types:
            raise ValueError("sentence_types must not be empty")

        for sentence_type in sentence_types:
            if not (isinstance(sentence_type, type) and issubclass(sentence_type, Sentence)):
                raise TypeError(f"{sentence_type} is not a Sentence subclass")

        self.code: str = code
        self.name: str = name
        self.sentence_types: tuple[type[Sentence], ...] = sentence_types
        self._get_instructions = get_instructions

    def get_instructions(self) -> str | None:
        """Return natural language grammar instructions for this language.

        Language packages should provide a callable via the get_instructions
        constructor parameter that returns vocabulary, grammar rules, and
        examples as a text prompt suitable for an LLM system message.

        Returns:
            Instructions string, or None if not provided.
        """
        if self._get_instructions:
            return self._get_instructions()
        return None

    def __repr__(self) -> str:
        """Return a string representation of the Language."""
        return f"Language(code={self.code!r}, types={len(self.sentence_types)})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on code."""
        if not isinstance(other, Language):
            return NotImplemented
        return self.code == other.code

    def __hash__(self) -> int:
        """Make Language hashable."""
        return hash(self.code)
