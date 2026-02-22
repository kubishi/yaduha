import time
from typing import ClassVar

from yaduha.agent import Agent
from yaduha.translator import Translation, Translator


class InstructionsTranslator(Translator):
    name: ClassVar[str] = "instructions_translator"
    description: ClassVar[str] = "Translate text to the target language and back to the source language using instructions."
    agent: Agent
    instructions: str

    def translate(self, text: str) -> Translation:
        start_time = time.time()
        response = self.agent.get_response(
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": text}
            ]
        )
        translation_time = time.time() - start_time

        self.logger.log(data={
            "event": "translation_complete",
            "translator": self.name,
            "source": text,
            "target": response.content,
            "translation_time": translation_time,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        })

        return Translation(
            source=text,
            target=response.content,
            translation_time=translation_time,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            back_translation=None,
            metadata={}
        )
