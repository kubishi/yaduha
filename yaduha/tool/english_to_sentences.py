import functools
import operator
import random
from typing import ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, create_model

from yaduha.agent import Agent, AgentResponse
from yaduha.language import Sentence
from yaduha.logger import inject_logs
from yaduha.tool import Tool

TSentenceType = TypeVar("TSentenceType", bound=Sentence)


class SentenceList(BaseModel, Generic[TSentenceType]):
    sentences: list[TSentenceType]


class EnglishToSentencesTool(Tool[AgentResponse[SentenceList[TSentenceType]]]):
    agent: "Agent"
    name: ClassVar[str] = "english_to_sentences"
    description: ClassVar[str] = "Translate natural English into a structured sentence."
    SentenceType: type[TSentenceType] | tuple[type[Sentence], ...]

    def _run(self, english: str) -> AgentResponse[SentenceList[TSentenceType]]:
        with inject_logs(tool="english_to_sentences"):
            # Handle both single type and union of types
            if isinstance(self.SentenceType, tuple):
                # Create a discriminated union type for multiple sentence types
                if len(self.SentenceType) == 1:
                    sentence_union = self.SentenceType[0]
                else:
                    sentence_union = functools.reduce(operator.or_, self.SentenceType)

                TargetSentenceList = create_model(
                    "TargetSentenceList", sentences=(list[sentence_union], ...), __base__=BaseModel
                )
            else:
                # Single sentence type (backward compatible)
                TargetSentenceList = create_model(
                    "TargetSentenceList",
                    sentences=(list[self.SentenceType], ...),
                    __base__=SentenceList[self.SentenceType],
                )

            examples = []
            SentenceTypes = (
                self.SentenceType if isinstance(self.SentenceType, tuple) else (self.SentenceType,)
            )
            for SentenceType in SentenceTypes:
                for example_english, example_sentence in SentenceType.get_examples():
                    examples.extend(
                        [
                            {"role": "user", "content": example_english},
                            {
                                "role": "assistant",
                                "content": SentenceList[TSentenceType](
                                    sentences=[example_sentence]
                                ).model_dump_json(),
                            },
                        ]
                    )

            response = self.agent.get_response(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a translator that transforms natural English sentences into structured sentences. "
                            "Given the output format, you may not be able to represent all the details of the input sentence, "
                            "but you must capture as much as meaning as possible. "
                        ),
                    },
                    *examples,
                    {"role": "user", "content": english},
                ],
                response_format=TargetSentenceList,
            )

            self.log(
                data={
                    "event": "english_to_sentences",
                    "content": english,
                    "response": response.content.model_dump_json(),
                    "response_time": response.response_time,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                }
            )

        return cast(AgentResponse[SentenceList[TSentenceType]], response)

    def get_examples(
        self,
    ) -> list[tuple[dict[str, str], AgentResponse[SentenceList[TSentenceType]]]]:
        examples = []
        if isinstance(self.SentenceType, tuple):
            sentence_types = self.SentenceType
        else:
            sentence_types = (self.SentenceType,)
        for SentenceType in sentence_types:
            for english, example_sentence in SentenceType.get_examples():
                examples.append(
                    (
                        {"english": english},
                        AgentResponse[SentenceList[TSentenceType]](
                            content=SentenceList[TSentenceType](sentences=[example_sentence]),
                            response_time=random.uniform(0.1, 0.5),
                            prompt_tokens=random.randint(10, 400),
                            completion_tokens=random.randint(10, 50),
                        ),
                    )
                )
        return examples
