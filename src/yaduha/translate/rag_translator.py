import json
import time

from yaduha.translate.base import Translation, Translator
from yaduha.chatbot import translate

class RAGTranslator(Translator):
    def __init__(self, model: str):
        self.model = model

    def translate(self, sentence: str) -> Translation:
        start = time.time()
        response = translate(sentence, model=self.model)
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
            metadata={
                "messages": json.dumps(response["messages"], ensure_ascii=False),
                "model": self.model,
            }
        )

        return translation