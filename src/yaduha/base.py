from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import openai
import dotenv
import os

from .sentence_builder import format_sentence, get_all_choices, print_sentence

dotenv.load_dotenv()

@dataclass
class Translation:
    source: str
    target: str
    back_translation: str

    translation_prompt_tokens: int
    translation_completion_tokens: int
    translation_time: float
    back_translation_prompt_tokens: int
    back_translation_completion_tokens: int
    back_translation_time: float

    def __str__(self):
        lines = [
            f"Source: {self.source}",
            f"Target: {self.target}",
            f"Back Translation: {self.back_translation}",
            f"Prompt Tokens: {self.translation_prompt_tokens}",
            f"Completion Tokens: {self.translation_completion_tokens}",
            f"Translation Time: {self.translation_time:0.2f}s",
            f"Back Translation Prompt Tokens: {self.back_translation_prompt_tokens}",
            f"Back Translation Completion Tokens: {self.back_translation_completion_tokens}",
            f"Back Translation Time: {self.back_translation_time:0.2f}s"
        ]
        return "\n".join(lines)
    
    def __repr__(self):
        return str(self)
    
    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "back_translation": self.back_translation,
            "translation_prompt_tokens": self.translation_prompt_tokens,
            "translation_completion_tokens": self.translation_completion_tokens,
            "translation_time": self.translation_time,
            "back_translation_prompt_tokens": self.back_translation_prompt_tokens,
            "back_translation_completion_tokens": self.back_translation_completion_tokens,
            "back_translation_time": self.back_translation_time,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            source=d["source"],
            target=d["target"],
            back_translation=d["back_translation"],
            translation_prompt_tokens=d.get("translation_prompt_tokens", 0),
            translation_completion_tokens=d.get("translation_completion_tokens", 0),
            translation_time=d.get("translation_time", 0.0),
            back_translation_prompt_tokens=d.get("back_translation_prompt_tokens", 0),
            back_translation_completion_tokens=d.get("back_translation_completion_tokens", 0),
            back_translation_time=d.get("back_translation_time", 0.0),
        )

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



class PipelineTranslator(Translator):
    pass

class PromptEngineeredTranslator(Translator):
    pass


    
