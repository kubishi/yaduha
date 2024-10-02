import argparse
from functools import lru_cache
from typing import Any, Dict, Optional, Set, Tuple
import openai
import dotenv
import os
import json
from enum import Enum
import pathlib
import shutil

from sentence_builder import format_sentence, get_all_choices, print_sentence
from translate_ovp2eng import translate

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

savedir = thisdir / ".logs" / "agentic"
# remove directory if it exists
if savedir.exists():
    shutil.rmtree(savedir)

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
    dictionary = {} # paiute to english
    r_dictionary = {} # english to paiute
    for part_of_speech, details in choices.items():
        dictionary[part_of_speech] = {ovp: eng for ovp, eng in details["choices"].items()}
        r_dictionary[part_of_speech] = {eng: ovp for ovp, eng in details["choices"].items()}

    try:
        word_choices = { # get english word choices
            pos: None if details['value'] is None else dictionary[pos][details['value']]
            for pos, details in choices.items()
        }
    except KeyError:
        for pos, details in choices.items():
            if details['value'] is not None and not (details['value'] in dictionary[pos]):
                print(f"Error: {pos} {details['value']} not in {dictionary[pos]}")
        raise

    enums = {}
    for part_of_speech, details in choices.items():
        enums[part_of_speech] = Enum(
            snake_to_pascal(part_of_speech),
            {"NULL": None, **{v: v for _, v in details["choices"].items()}},
        )

    options = {
        part_of_speech: (
            enums[part_of_speech], # if details["requirement"] == "required" else Optional[enums[part_of_speech]],
            (
                (... if details["requirement"] == "required" else None) if not details["value"] 
                else enums[part_of_speech](dictionary[part_of_speech][details["value"]])
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
    word_choices = {}
    target_sentence = None
    iteration, max_iterations = 0, 10
    while True:
        iteration += 1
        Choice, word_choices, r_dictionary = get_choice_model(word_choices)
        savepath = savedir / "schemas" / f"choice_{iteration}.json"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        savepath.write_text(json.dumps(Choice.model_json_schema(), indent=2, ensure_ascii=False))
        
        
        message = json.dumps(
            {
                "sentence": sentence,
                "target_sentence": target_sentence,
                "can_terminate": target_sentence is not None
            },
            ensure_ascii=False
        )
        print(Choice.schema_json(indent=2))
        print(message)
        print(word_choices)
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
        savepath = savedir / "messages" / f"message_{iteration}.json"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        savepath.write_text(json.dumps(messages, indent=2, ensure_ascii=False))
        res = client.beta.chat.completions.parse(
            model="gpt-4o",
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str, help="The sentence to build.")

    args = parser.parse_args()

    word_choices = build_sentence(args.sentence)
    structured_sentence = format_sentence(**word_choices)
    translation = translate(**word_choices)
    print_sentence(structured_sentence)
    print(translation)

if __name__ == "__main__":
    main()