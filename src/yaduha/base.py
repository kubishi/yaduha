from abc import ABC, abstractmethod
from typing import Tuple
import openai
import dotenv
import os

from .sentence_builder import format_sentence, get_all_choices, print_sentence

dotenv.load_dotenv()


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> Tuple[str, str]:
        """Translate the text to the target language and back to the source language.
        
        Args:
            text (str): The text to translate.

        Returns:
            Tuple[str, str]: The translated text and the back-translated text respectively.
        """
        raise NotImplementedError



class PipelineTranslator(Translator):
    pass

class PromptEngineeredTranslator(Translator):
    pass


    
