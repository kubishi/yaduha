"""Exceptions for language loading and validation."""


class LanguageNotFoundError(Exception):
    """Raised when a language cannot be found or loaded."""

    pass


class LanguageValidationError(Exception):
    """Raised when a language fails validation checks."""

    pass
