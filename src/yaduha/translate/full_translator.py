import json
import time
import logging

from yaduha.common import get_openai_client
from yaduha.translate.base import Translation, Translator
from yaduha.chatbot.tools.functions import search_english, search_sentences
from yaduha.translate.pipeline import split_sentence
from yaduha.translate.pipeline import translate_simple, order_sentence
from yaduha.translate.full_translator_tools import tools, translation_messages
from openai.types.chat import ChatCompletion

client = get_openai_client()

def translate_simple_sentences(sentence: str, model: str = "gpt-4o-mini"):
    
    simple_sentences = split_sentence(sentence, model)

    target_simple_sentences = []

    for sentence in simple_sentences.sentences:
        subject, verb, _object = translate_simple(sentence)
        target_simple_sentence = order_sentence(subject, verb, _object)
        target_simple_sentences.append(" ".join(map(str, target_simple_sentence)))
    
    target_simple_sentence_nl = ". ".join(target_simple_sentences) + '.'
    return target_simple_sentence_nl
    

functions = {
    "search_english": search_english,
    "search_sentences": search_sentences,
    "split_sentence": split_sentence,
    "translate_simple_sentence": translate_simple_sentences
}

def translate_sentence(sentence: str, model: str = "gpt-4o-mini"):
    messages = [
        *translation_messages,
        {
            "role": "user",
            "content": sentence
        }
    ]

    while True:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            tools=tools,
        )

        messages.append(json.loads(completion.choices[0].message.model_dump_json()))

        if completion.choices[0].message.content:
            logging.info("Response: " + completion.choices[0].message.content)

        if not completion.choices[0].message.tool_calls:
            translation = completion.choices[0].message.content
            break
        
        for tool_call in completion.choices[0].message.tool_calls:
            kwargs = json.loads(tool_call.function.arguments)
            logging.info(f"Function: {tool_call.function.name}")
            # logging.info(f"Arguments: {kwargs}")
            function = functions.get(tool_call.function.name)
            if not function:
                logging.error(f"Function {tool_call.function.name} not found.")
                continue
            res = function(**kwargs)
            # logging.info(f"Result: {res}")
            messages.append({
                "role": "tool",
                "content": json.dumps(res, ensure_ascii=False),
                "tool_call_id": tool_call.id
            })

    response = {
        "translation_prompt_tokens": completion.usage.prompt_tokens,
        "translation_completion_tokens": completion.usage.completion_tokens,
        "translation_total_tokens": completion.usage.total_tokens,
        "translation": translation,
        "messages": messages
    }
    return response

class FullTranslator(Translator):
    def __init__(self, model: str):
        self.model = model

    def translate(self, sentence: str) -> Translation:
        start = time.time()
        response = translate_sentence(sentence, model=self.model)
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