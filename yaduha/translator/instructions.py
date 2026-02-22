import time
from collections.abc import Sequence
from typing import ClassVar

from pydantic import Field

from yaduha.agent import Agent
from yaduha.evaluator import Evaluator
from yaduha.translator import Translation, Translator
from yaduha.translator.back_translator import BackTranslator


class InstructionsTranslator(Translator):
    name: ClassVar[str] = "instructions_translator"
    description: ClassVar[str] = (
        "Translate text to the target language and back to the source language using instructions."
    )
    agent: Agent
    instructions: str
    back_translator: BackTranslator | None = None
    evaluators: Sequence[Evaluator] = Field(default_factory=list)

    def translate(self, text: str) -> Translation:
        start_time = time.time()
        response = self.agent.get_response(
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": text},
            ]
        )
        translation_time = time.time() - start_time

        target = response.content

        back_translation = None
        if self.back_translator:
            back_translation = self.back_translator.back_translate(target)

        evaluations = {}
        if back_translation and self.evaluators:
            evaluations = {
                e.name: e.evaluate(text, back_translation.source) for e in self.evaluators
            }

        self.logger.log(
            data={
                "event": "translation_complete",
                "translator": self.name,
                "source": text,
                "target": target,
                "back_translation": back_translation.source if back_translation else None,
                "translation_time": translation_time,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "evaluations": evaluations,
            }
        )

        return Translation(
            source=text,
            target=target,
            translation_time=translation_time,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            back_translation=back_translation,
            evaluations=evaluations,
        )
