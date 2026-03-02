import time
from collections.abc import Sequence
from typing import ClassVar

from pydantic import Field

from yaduha.agent import Agent
from yaduha.evaluator import Evaluator
from yaduha.loader import LanguageLoader
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

    @classmethod
    def from_language(
        cls,
        language_code: str,
        agent: Agent,
        back_translator: BackTranslator | None = None,
        evaluators: Sequence[Evaluator] | None = None,
        **kwargs,
    ) -> "InstructionsTranslator":
        """Create an InstructionsTranslator from an installed language package.

        Args:
            language_code: Language code (e.g., 'ovp')
            agent: Agent to use for translation
            back_translator: Optional back-translator for verification
            evaluators: Optional list of evaluators for translation quality

        Returns:
            InstructionsTranslator instance

        Raises:
            ValueError: If the language does not provide instructions
        """
        language = LanguageLoader.load_language(language_code)
        instructions = language.get_instructions()
        if not instructions:
            raise ValueError(
                f"Language '{language_code}' does not provide instructions. "
                "Pass a get_instructions callable to the Language constructor."
            )
        return cls(
            agent=agent,
            instructions=instructions,
            back_translator=back_translator,
            evaluators=evaluators or [],
            **kwargs,
        )

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
                "back_translation_time": back_translation.translation_time
                if back_translation
                else None,
                "back_translation_prompt_tokens": back_translation.prompt_tokens
                if back_translation
                else None,
                "back_translation_completion_tokens": back_translation.completion_tokens
                if back_translation
                else None,
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
