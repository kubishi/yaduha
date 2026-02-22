import time
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field

from yaduha.agent import Agent
from yaduha.tool import Tool
from yaduha.translator import Translation, Translator


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceItem(BaseModel):
    tool_name: str = Field(..., description="The name of the tool used.")
    tool_input: str = Field(..., description="The input provided to the tool.")
    tool_output: str = Field(..., description="The output returned by the tool.")


class TranslationResponse(BaseModel):
    translation: str = Field(..., description="The translated text.")
    confidence: ConfidenceLevel = Field(..., description="The confidence level of the translation.")
    evidence: list[EvidenceItem] = Field(..., description="The evidence used in the translation.")


class AgenticTranslator(Translator):
    name: ClassVar[str] = "agentic_translator"
    description: ClassVar[str] = "Translate text using an agentic approach."

    agent: Agent
    system_prompt: str = Field(
        default=(
            "You are a translation agent that uses tools to translate text accurately. "
            "Use the tools available to you as needed to produce the best translation possible. "
            "You can use one or many tool calls (in parallel and/or sequentially) until you decide to respond. "
            "Only respond with the final translation, your confidence level, and the evidence used in your translation."
        ),
        description="The system prompt to guide the agent's behavior.",
    )
    tools: list[Tool] | None = Field(
        None,
        description="A list of tools that the agent can use for translation.",
    )

    def translate(self, text: str) -> Translation:
        start_time = time.time()
        response = self.agent.get_response(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Translate the following text:\n\n{text}",
                },
            ],
            response_format=TranslationResponse,
            tools=self.tools,
        )
        end_time = time.time()

        confidence = response.content.confidence
        evidence = [item.model_dump() for item in response.content.evidence]

        self.logger.log(
            data={
                "event": "translation_complete",
                "translator": self.name,
                "source": text,
                "target": response.content.translation,
                "translation_time": end_time - start_time,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "confidence": confidence.value,
                "evidence": evidence,
                "num_tool_calls": len(evidence),
            }
        )

        translation = Translation(
            source=text,
            target=response.content.translation,
            translation_time=end_time - start_time,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            back_translation=None,
        )

        return translation
