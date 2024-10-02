import argparse
from functools import lru_cache
import os
from typing import Any, Dict, Optional, Set, Tuple
import openai
import dotenv
import json
from enum import Enum

from .sentence_builder import format_sentence, get_all_choices, print_sentence, sentence_to_str
from .base import Translator
from .translate_ovp2eng import translate as translate_ovp2eng

dotenv.load_dotenv()

from pydantic import BaseModel, create_model

def snake_to_pascal(snake_str):
    components = snake_str.split('_')
    pascal_str = ''.join(x.capitalize() for x in components)    
    return pascal_str

@lru_cache(maxsize=None)
def get_parts_of_speech() -> Set[str]:
    return get_all_choices().keys()

def get_choice_model(word_choices: Dict[str, Optional[str]]) -> Tuple[BaseModel, Dict[str, Optional[str]], Dict[str, Dict[str, str]]]:
    choices = get_all_choices(**word_choices)
    r_dictionary = {} # english to paiute
    for part_of_speech, details in choices.items():
        r_dictionary[part_of_speech] = {eng: ovp for ovp, eng in details["choices"].items()}

    try:
        word_choices = { # get english word choices
            pos: None if details['value'] is None else choices[pos]['choices'][details['value']]
            for pos, details in choices.items()
        }
    except KeyError:
        for pos, details in choices.items():
            if details['value'] is not None and not (details['value'] in choices[pos]['choices']):
                print(f"Error: {pos} {details['value']} not in {choices[pos]['choices']}")
        raise

    enums = {}
    for part_of_speech, details in choices.items():
        enums[part_of_speech] = Enum(
            snake_to_pascal(part_of_speech),
            {v: v for _, v in details["choices"].items()}
            # (
            #     {"NULL": None, **{v: v for _, v in details["choices"].items()}}
            #     if part_of_speech in ["subject_noun", "object_noun", "verb", "adjective", "adverb"]
            #     else {v: v for _, v in details["choices"].items()}
            # )
        )

    options = {
        part_of_speech: (
            enums[part_of_speech], # if details["requirement"] == "required" else Optional[enums[part_of_speech]],
            (
                (... if details["requirement"] == "required" else None) if not details["value"] 
                else enums[part_of_speech](choices[part_of_speech]['choices'][details["value"]])
            ),
        )
        for part_of_speech, details in choices.items()
        if details["requirement"] != "disabled"
    }
    Choice = create_model(
        "Choice",
        terminate=(bool, ...),
        **options
    )

    return Choice, word_choices, r_dictionary

def build_sentence(sentence: str) -> Dict[str, Optional[str]]:
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    
    word_choices = {}
    target_sentence = None
    iteration, max_iterations = 0, 10
    while True:
        iteration += 1
        Choice, word_choices, r_dictionary = get_choice_model(word_choices)   

        
        message = json.dumps(
            {
                "sentence": sentence,
                "target_sentence": target_sentence,
                "can_terminate": target_sentence is not None
            },
            ensure_ascii=False
        ) 
        print(Choice.schema_json(indent=2))
        print(word_choices)  
        print(message)  
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agent that uses available tools to build sentences that most closely approximate the meaning of user input sentences. "
                    "Some of the word choices may have already been made for you (value is not None). You should use these to inform your next choices. "
                    "If the sentence is complete, please select the 'Terminate' option. "
                    "You can only terminate if the target sentence is valid (not None). "
                    "You may only select words for the available parts of speech and vocabulary. "
                    "This may mean the target sentence is different from the input sentence (missing words, etc.). "
                    "Always prioritize selecting a required part of speech next. "
                    "Only select a non-required part of speech if all required parts of speech have been selected. "
                )
            },
            {
                "role": "user",
                "content": message,
            }
        ]
        res = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=Choice,
            temperature=0.0
        )
        
        choice = res.choices[0].message.parsed
        for part_of_speech in get_parts_of_speech():
            if hasattr(choice, part_of_speech):
                word = getattr(choice, part_of_speech)
                if word is not None and word.value is not None:
                    word_choices[part_of_speech] = r_dictionary[part_of_speech][word.value]

        try:
            target_sentence = format_sentence(**word_choices)
        except:
            target_sentence = None

        if choice.terminate:
            return word_choices
        
        if iteration >= max_iterations:
            raise Exception("Max iterations reached")


class AgenticTranslator(Translator):
    """Translator that takes an agentive approach to translation.
    
    In each step, the translator uses the 
    """
    
    def __init__(self, model: str):
        self.client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

    def translate(self, text: str) -> Tuple[str, str]:
        # choices = get_all_choices()
        sentences = []
        tools = {
            "build_sentence": {
                "type": "function",
                "function": {
                    "name": "build_sentence",
                    "description": "Attempt to build a simple sentence in Paiute.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sentence": {"type": "string"}
                        },
                        "required": ["sentence"],
                        "additionalProperties": False
                    }
                }
            }
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agent that uses available tools to provide translations of user input sentences. "
                    "You can use the 'build_sentence' tool to attempt to build a simple sentence in Paiute. "
                    "You can call the tool multiple times to build different sentences and then decide what to return to the user. "
                )
            },
            {
                "role": "user",
                "content": text,
            }
        ]

        while True:
            print()
            print(messages)
            res = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[tools["build_sentence"]],
                temperature=0.0,
                # tool_choice={"type": "function", "function": {"name": "build_sentence"}}
            )

            messages.append(json.loads(res.choices[0].message.model_dump_json()))

            if res.choices[0].message.tool_calls:
                for tool_call in res.choices[0].message.tool_calls:
                    if tool_call.function.name == "build_sentence":
                        args = json.loads(tool_call.function.arguments)
                        try:
                            word_choices = build_sentence(args['sentence'])
                            structured_sentence = format_sentence(**word_choices)
                            target = sentence_to_str(structured_sentence)
                            english = translate_ovp2eng(**word_choices)
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"success": True, "target": target, "english": english}),
                                "tool_call_id": tool_call.id
                            })
                        except Exception as e:
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"success": False, "error": str(e)}),
                                "tool_call_id": tool_call.id
                            })
                            # raise e

            if res.choices[0].message.content:
                return res.choices[0].message.content



        