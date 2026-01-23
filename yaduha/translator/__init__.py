from abc import ABC, abstractmethod
from typing import ClassVar
from uuid import uuid4
from pydantic import BaseModel, Field
from yaduha.logger import inject_logs
from yaduha.tool import Tool


class BackTranslation(BaseModel):
    source: str = Field(..., description="The back translated source-language text.")
    target: str = Field(..., description="The original target-language text.")
    translation_time: float = Field(..., description="The time taken for back translation.")
    prompt_tokens: int = Field(0, description="The number of prompt tokens used for back translation.")
    completion_tokens: int = Field(0, description="The number of completion tokens used for back translation.")

class Translation(BaseModel):
    source: str = Field(..., description="The source-language text.")
    target: str = Field(..., description="The target-language text.")
    translation_time: float = Field(..., description="The time taken for translation.")
    prompt_tokens: int = Field(0, description="The number of prompt tokens used for the entire translation.")
    completion_tokens: int = Field(0, description="The number of completion tokens used for the entire translation.")
    back_translation: BackTranslation | None = Field(
        None, description="The back translation details, if available."
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the translation.")

class Translator(Tool[Translation], ABC):
    """Base class for translators that translate text to a target language and back to the source language."""
    name: ClassVar[str] = "translator"
    description: ClassVar[str] = "Translate text to the target language and back to the source language."

    def _run(self, text: str) -> Translation:
        """Translate the text to the target language and back to the source language.

        Args:
            text (str): The text to translate.
        Returns:
            Translation: The translation
        """
        return self.translate(text)   

    @abstractmethod
    def translate(self, text: str) -> Translation:
        pass
