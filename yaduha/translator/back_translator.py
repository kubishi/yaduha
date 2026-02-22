from abc import abstractmethod
from typing import ClassVar

from yaduha.tool import Tool
from yaduha.translator import BackTranslation


class BackTranslator(Tool[BackTranslation]):
    """Base class for back-translators that translate target-language text back to the source language."""

    name: ClassVar[str] = "back_translator"
    description: ClassVar[str] = (
        "Translate text from the target language back to the source language."
    )

    def _run(self, text: str) -> BackTranslation:
        return self.back_translate(text)

    @abstractmethod
    def back_translate(self, text: str) -> BackTranslation:
        pass
