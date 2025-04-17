import time
import os

from yaduha.base import Translation, Translator
from yaduha.chatbot import translate

class RAGTranslator(Translator):
    def __init__(self):
        pass

    def translate(self, sentence: str) -> Translation:
        start = time.time()
        response = translate(sentence)
        end = time.time()
        time_taken = end - start
        translation = Translation(
            source=sentence,
            target=response["translation"],
            back_translation="",
            translation_prompt_tokens=response["translation_prompt_tokens"],
            translation_completion_tokens=response["translation_completion_tokens"],
            translation_total_tokens=response["translation_total_tokens"],
            translation_time=time_taken,
            back_translation_prompt_tokens=0,
            back_translation_completion_tokens=0,
            back_translation_total_tokens=0,
            back_translation_time=0.0,
        )

        return translation