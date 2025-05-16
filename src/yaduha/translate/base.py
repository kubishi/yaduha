from abc import ABC, abstractmethod
from typing import Dict, Union
import dotenv
from pydantic import BaseModel

dotenv.load_dotenv()

class Translation(BaseModel):
    source: str
    target: str
    back_translation: str

    translation_prompt_tokens: int
    translation_completion_tokens: int
    translation_time: float
    back_translation_prompt_tokens: int
    back_translation_completion_tokens: int
    back_translation_time: float

    metadata: Dict[str, Union[str, int, float]] = {}

    def __str__( self ) -> str:
        lines = [
            f"Source: {self.source}",
            f"Target: {self.target}",
            f"Back Translation: {self.back_translation}",
            f"Translation Prompt Tokens: {self.translation_prompt_tokens}",
            f"Translation Completion Tokens: {self.translation_completion_tokens}",
            f"Translation Time: {self.translation_time:.2f} seconds",
            f"Back Translation Prompt Tokens: {self.back_translation_prompt_tokens}",
            f"Back Translation Completion Tokens: {self.back_translation_completion_tokens}",
            f"Back Translation Time: {self.back_translation_time:.2f} seconds",
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.__str__()
    

class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> Translation:
        """Translate the text to the target language and back to the source language.
        
        Args:
            text (str): The text to translate.

        Returns:
            Translation: The translation
        """
        raise NotImplementedError


    
