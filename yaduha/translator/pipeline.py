import random
import re
import time
from uuid import uuid4
from typing import ClassVar, Dict, Generic, List, Optional, Type, Tuple

from yaduha import logger
from yaduha.logger import inject_logs
from yaduha.translator import Translator, Translation, BackTranslation
from yaduha.tool.english_to_sentences import EnglishToSentencesTool, TSentenceType
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha.agent import Agent
from yaduha.language import Sentence

class PipelineTranslator(Translator, Generic[TSentenceType]):
    name: ClassVar[str] = "pipeline_translator"
    description: ClassVar[str] = (
        "Translate text use a model of the target language. "
        "This approach guarantees grammatical output in the target language but may lose some meaning "
        "from the input text due to limitations in the sentence structures available in the target language."
    )

    agent: Agent
    back_translation_agent: Optional[Agent] = None
    SentenceType: Type[TSentenceType] | Tuple[Type[Sentence], ...]

    def translate(self, text: str) -> Translation:
        """Translate the text using a pipeline of translators.
        
        Args:
            text (str): The text to translate.
        Returns:
            Translation: The translation
        """
        start_time = time.time()
        translate_input_to_sentences = EnglishToSentencesTool(
            agent=self.agent,
            SentenceType=self.SentenceType,
            logger=self.logger
        )
        # Use back_translation_agent if provided, otherwise fall back to main agent
        bt_agent = self.back_translation_agent or self.agent
        translate_sentence_to_english = SentenceToEnglishTool(
            agent=bt_agent,
            SentenceType=self.SentenceType,
            logger=self.logger
        )

        def clean_text(s: str) -> str:
            s = s.strip()
            # add a period if it doesn't end with punctuation
            if not re.search(r'[.!?]$', s):
                s += '.'
            # capitalize the first letter
            s = s[0].upper() + s[1:]
            return s

        sentences_response = translate_input_to_sentences(text)
        end_time = time.time()

        targets = []
        back_translations = []
        prompt_tokens = sentences_response.prompt_tokens
        completion_tokens = sentences_response.completion_tokens
        prompt_tokens_bt = 0
        completion_tokens_bt = 0

        start_time_bt = time.time()
        for sentence in sentences_response.content.sentences:
            targets.append(clean_text(str(sentence)))
            back_translation = translate_sentence_to_english(sentence)
            back_translations.append(clean_text(back_translation.content))
            prompt_tokens_bt += back_translation.prompt_tokens
            completion_tokens_bt += back_translation.completion_tokens
        end_time_bt = time.time()

        if self.back_translation_agent:
                
            self.logger.log(data={
                "event": "translation_completed",
                "agent_name": self.agent.name,
                "agent_model": self.agent.model,
                "back_translation_agent_name": self.back_translation_agent.name,
                "back_translation_agent_model": self.back_translation_agent.model,
                "response": " ".join(targets), 
                "source": text,
                "translation_time": end_time - start_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "back_translation_time": end_time_bt - start_time_bt,
                "back_translation_prompt_tokens": prompt_tokens_bt,
                "back_translation_completion_tokens": completion_tokens_bt
            })
        else:
            self.logger.log(data={
                "event": "translation_completed",
                "agent_name": self.agent.name,
                "agent_model": self.agent.model,
                "response": " ".join(targets), 
                "source": text,
                "translation_time": end_time - start_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "back_translation_time": end_time_bt - start_time_bt,
                "back_translation_prompt_tokens": prompt_tokens_bt,
                "back_translation_completion_tokens": completion_tokens_bt
            })

        return Translation(
            source=text,
            target=" ".join(targets),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            translation_time=end_time - start_time,
            back_translation=BackTranslation(
                source=" ".join(back_translations),
                target=" ".join(targets),
                prompt_tokens=prompt_tokens_bt,
                completion_tokens=completion_tokens_bt,
                translation_time=end_time_bt - start_time_bt
            ),
        )

    def get_examples(self) -> List[Tuple[Dict[str, str], Translation]]:
        examples = []
        translate_input_to_sentences = EnglishToSentencesTool(
            agent=self.agent,
            SentenceType=self.SentenceType
        )
        bt_agent = self.back_translation_agent or self.agent
        translate_sentence_to_english = SentenceToEnglishTool(
            agent=bt_agent,
            SentenceType=self.SentenceType
        )

        for input_example, sentence_list in translate_input_to_sentences.get_examples():
            targets = []
            back_translations = []
            for sentence in sentence_list.content.sentences:
                targets.append(str(sentence))
                _, english_response = translate_sentence_to_english.get_examples()[0]
                back_translations.append(english_response.content)

            text_input = input_example["english"]
            translation = Translation(
                source=text_input,
                target=" ".join(targets),
                prompt_tokens=random.randint(10, 300),
                completion_tokens=random.randint(10, 100),
                translation_time=random.uniform(0.5, 2.0),
                back_translation=BackTranslation(
                    source=" ".join(back_translations),
                    target=" ".join(targets),
                    prompt_tokens=random.randint(10, 300),
                    completion_tokens=random.randint(10, 100),
                    translation_time=random.uniform(0.5, 2.0)
                ),
            )
            examples.append(({"text": text_input}, translation))
        return examples