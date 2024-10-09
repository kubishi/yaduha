import time
from ..base import Translation, Translator
import openai
import os

class FinetunedTranslator(Translator):
    def __init__(self, model: str):
        self.model = model
        self.client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

    def translate(self, sentence: str) -> Translation:
        start_time = time.time()
        messages = [
            {
                "role": "system",
                "content": (
                    "Translate the following text from English to OVP. "
                    "For any word that does not have an equivalent in OVP, "
                    "leave the word untranslated and place it inside brackets. "
                    "Do not attempt to translate words inside brackets. "
                )
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Sleet climbs that rafter."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                    "type": "text",
                    "text": "[sleet]-uu [rafter]-noka u-dsibui-dü"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "The pouch has smelled this ledge."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                    "type": "text",
                    "text": "[pouch]-uu [ledge]-neika a-gwana-pü"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "The jackrabbit is eating the bread."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                    "type": "text",
                    "text": "kamü-uu tüwoobü-neika a-düka-ti "
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": sentence
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        target = response.choices[0].message.content
        translation_time = time.time() - start_time
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        translation = Translation(
            source=sentence,
            target=target,
            back_translation="",
            translation_prompt_tokens=prompt_tokens,
            translation_completion_tokens=completion_tokens,
            translation_time=translation_time,
            back_translation_prompt_tokens=0,
            back_translation_completion_tokens=0,
            back_translation_time=0
        )
        return translation
