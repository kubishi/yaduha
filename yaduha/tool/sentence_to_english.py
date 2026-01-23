

import json
from typing import ClassVar, Generic, List, Tuple, Type, Dict, cast
from yaduha.language import Sentence

from yaduha.logger import inject_logs
from yaduha.tool import Tool
from yaduha.agent import Agent, AgentResponse
from yaduha.tool.english_to_sentences import TSentenceType

class SentenceToEnglishTool(Tool[AgentResponse[str]], Generic[TSentenceType]):
    agent: Agent
    name: ClassVar[str] = "sentence_to_english"
    description: ClassVar[str] = "Translate a structured sentence into natural English."
    SentenceType: Type[TSentenceType] | Tuple[Type[Sentence], ...]

    def _run(self, sentence: TSentenceType) -> AgentResponse:
        with inject_logs(tool="sentence_to_english"):
            example_messages = []
            for english, example_sentence in cast(List[Tuple[str, TSentenceType]], sentence.get_examples()):
                example_messages.append({
                    "role": "user",
                    "content": json.dumps(example_sentence.model_dump_json(), ensure_ascii=False)
                })
                example_messages.append({
                    "role": "assistant",
                    "content": english
                })

            response = self.agent.get_response(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a translator that transforms structured sentences into natural English. "
                            "The sentences may be strange and unusual, but you must translate them as accurately as possible. "
                        )
                    },
                    *example_messages,
                    {
                        "role": "user",
                        "content": json.dumps(sentence.model_dump_json(), ensure_ascii=False)
                    }
                ]
            )

            self.log(data={
                "sentence": sentence.model_dump_json(),
                "response": response.content,
                "response_time": response.response_time,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            })

        return response

    def get_examples(self) -> List[Tuple[Dict[str, TSentenceType], AgentResponse]]:
        import random
        examples = []
        if isinstance(self.SentenceType, tuple):
            sentence_types = self.SentenceType
        else:
            sentence_types = (self.SentenceType,)
        for SentenceType in sentence_types:
            for english, example_sentence in SentenceType.get_examples():
                examples.append(
                    (
                        {"sentence": example_sentence},
                        AgentResponse(
                            content=english,
                            response_time=random.uniform(0.1, 0.5),
                            prompt_tokens=random.randint(10, 300),
                            completion_tokens=random.randint(10, 50)
                        )
                    )
                )
        return examples