"""Language class for wrapping sentence types and metadata."""

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
    ) -> None:
        """Initialize a Language instance.

        Args:
            code: Language code identifier
            name: Human-readable language name
            sentence_types: Tuple of Sentence subclasses

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
