import json
import logging
from typing import Tuple
import openai
import dotenv
import os
import pathlib

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

thisdir = pathlib.Path(__file__).parent.absolute()

VERBS = json.loads((thisdir / 'verbs.json').read_text())

NOUNS = json.loads((thisdir / 'nouns.json').read_text())

VERB_TENSES = {
    'present': 'dü',
    'present-ongoing': 'ti',
    'past': 'ku',
    'future': 'wei'
}

SUBJECT_PRONOUNS = {
    "first_person_singular": "nüü",	
    "first_person_plural_exclusive": "nüügwa",
    "first_person_plural_inclusive": "taagwa",
    "first_person_plural_inclusive_dual": "taa",
    "third_person_proximal": "mahu",
    "third_person_distal": "uhu",
    "third_person_plural_proximal": "mahuw̃a",
    "third_person_plural_distal": "uhuw̃a",
    "second_person_singular": "üü",
    "second_person_plural": "üügwa"
}

OBJECT_PRONOUNS = {
    "first_person_singular": "i",
    "first_person_plural_exclusive": "ni",
    "first_person_plural_inclusive": "tei",
    "first_person_plural_inclusive_dual": "ta",
    "third_person_proximal": "ma",
    "third_person_distal": "u",
    "third_person_plural_proximal": "mai",
    "third_person_plural_distal": "ui",
    "second_person_singular": "ü",
    "second_person_plural": "üi"
}

subject_pronoun_schema = {
    "name": "pronoun",
    "description": "A subject pronoun. (first_person_singular=I, first_person_plural_exclusive=we (exclusive), ...). Proximity and plurality may be ambiguous for some sentences - in this case, just pick one.",
    "type": "object",
    "properties": {
        "value": {            
            "type": "string",
            "enum": list(SUBJECT_PRONOUNS.keys())
        }
    },
    "required": ["value"]
}

object_pronoun_schema = {
    "name": "pronoun",
    "description": "An object pronoun. (first_person_singular=me, first_person_plural_exclusive=us, ...). Proximity and plurality may be ambiguous for some sentences - in this case, just pick one.",
    "type": "object",
    "properties": {
        "value": {
            "type": "string",
            "enum": list(OBJECT_PRONOUNS.keys())
        }
    },
    "required": ["value"]
}

noun_schema = {
    "name": "noun",
    "description": "A noun.",
    "type": "object",
    "properties": {
        "value": {
            "type": "string",
            "description": "The noun.",
            # "enum": list(NOUNS.keys())
        },
        "proximal": {
            "type": "boolean",	
            "description": "Whether the noun is proximal to the speaker. May be ambiguous for some sentences - in this case, just pick one."
        }
    },
    "required": ["value", "proximal"]
}

verb_stem_schema = {
    "name": "stem",
    "description": "A verb stem (i.e. eat, drink, sleep, run, etc.).",
    "type": "string",
    # "enum": list(VERBS.keys())
}

verb_tense_schema = {
    "name": "tense",
    "description": "A verb tense.",
    "type": "string",
    "enum": list(VERB_TENSES.keys())
}

sentence_schema = {
  "type": "object",
  "properties": {
    "subject": {
      "oneOf": [subject_pronoun_schema, noun_schema]
    },
    "verb": {
        "type": "object",
        "properties": {
            "stem": verb_stem_schema,
            "tense": verb_tense_schema
        },
        "required": ["stem", "tense"]
    },
    "object": {
        "oneOf": [object_pronoun_schema, noun_schema]
    }
  },
  "required": ["subject", "verb"]
}

LENIS_MAP = {
    'p': 'b',
    't': 'd',
    'k': 'g',
    's': 'z',
    'm': 'w̃'
}       
def to_lenis(word: str) -> str:
    """Convert a word to its lenis form
    """
    first_letter = word[0]
    if first_letter in LENIS_MAP:
        return LENIS_MAP[first_letter] + word[1:]
    else:
        return word

build_sentence_func = {
    "name": "build_sentence",
    "description": "Build a sentence from a subject, verb, and object.",
    "parameters": sentence_schema
}

def build_sentence(subject: dict, verb: dict, object: dict = None) -> str:
    subject_noun = subject["value"].lower().strip()
    subject_proximal = subject.get("proximal", False)

    verb_stem = verb["stem"].lower().strip()
    verb_tense = verb["tense"].lower().strip()

    object_noun, object_proximal = None, False
    if object is not None:
        object_noun = object["value"].lower().strip()
        object_proximal = object.get("proximal", False)

    if subject_noun in SUBJECT_PRONOUNS:
        subject_text = SUBJECT_PRONOUNS[subject_noun]
    else:
        subject_suffix = "ii" if subject_proximal else "uu"
        _subject_noun = NOUNS.get(subject_noun, f"[{subject_noun}]")
        subject_text = f"{_subject_noun}-{subject_suffix}"
    
    _verb_stem = VERBS.get(verb_stem, f"[{verb_stem}]")
    verb_text = f"{_verb_stem}-{VERB_TENSES[verb_tense]}"

    if isinstance(object_noun, str) and object_noun in OBJECT_PRONOUNS:
        _pronoun = OBJECT_PRONOUNS[object_noun]
        verb_text = f"{_pronoun}-{to_lenis(verb_text)}"
        object_text = ""
    elif object_noun is not None:
        object_suffix = "neika" if object_proximal else "noka"
        if object_noun.endswith("'"):
            object_suffix = object_suffix = "eika" if object_proximal else "oka"
        _object_noun = NOUNS.get(object_noun, f"[{object_noun}]")
        object_text = f"{_object_noun}-{object_suffix}"

        _pronoun = OBJECT_PRONOUNS["third_person_proximal"] if object_proximal else OBJECT_PRONOUNS["third_person_distal"]
        verb_text = f"{_pronoun}-{to_lenis(verb_text)}"
    else:
        object_text = ""

    if object_text:
        if subject_noun in SUBJECT_PRONOUNS:
            return f"{object_text} {subject_text} {verb_text}"
        else:
            return f"{subject_text} {object_text} {verb_text}"
    else:
        if subject_noun in SUBJECT_PRONOUNS:
            return f"{verb_text} {subject_text}"
        else:
            return f"{subject_text} {verb_text}"


def to_english(structured_sentence: dict) -> str:
    logging.info(structured_sentence)
    examples = [
        {
            "role": "user",
            "content": r"{'subject': {'value': 'first_person_singular'}, 'verb': {'stem': 'eat', 'tense': 'present-ongoing'}, 'object': {'value': 'fish', 'proximal': False}}"
        },
        {
            "role": "assistant",
            "content": "I am eating that fish."
        },
        {
            "role": "user",
            "content": r"{'subject': {'value': 'rice'}, 'verb': {'stem': 'drink', 'tense': 'future'}, 'object': {'value': 'jackrabbit', 'proximal': False}}",
        },
        {
            "role": "assistant",
            "content": "That rice will drink that jackrabbit."
        }
    ]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a bot that ranslates structured data into natural English."
            },
            *examples,
            {
                "role": "user",
                "content": json.dumps(structured_sentence)
            }
        ]
    )

    response_message = response["choices"][0]["message"]["content"]
    return response_message

def translate(query: str) -> Tuple[str, str]:
    examples = [
        {"role": "user", "content": "I like Beyonce."},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": "{\n  \"subject\": {\n    \"value\": \"first_person_singular\"\n  },\n  \"verb\": {\n    \"stem\": \"like\",\n    \"tense\": \"present\"\n  },\n  \"object\": {\n    \"value\": \"Beyonce\"\n  }\n}",
                "name": "build_sentence"
            },
        }
    ]
    
    messages = [
        {
            "role": "system",
            "content": (
                "Save the pieces of the sentences you are provided using the most appropriate available function. "
            )
        },
        *examples,
        {
            "role": "user",
            "content": query
        }
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        functions=[build_sentence_func],
        function_call={'name': 'build_sentence'}
    )

    response_message = response["choices"][0]["message"]
    messages.append(response_message)
    
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])

        logging.info(f"{function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")
        sentence = build_sentence(**function_args)

        sentence_data = {', '.join(f'{k}={v}' for k, v in function_args.items())}
        english = to_english(function_args)
        return english, sentence


def main():
    logging.basicConfig(level=logging.WARNING)

    while True:
        query = input("Enter an English sentence: ")
        english, paiute = translate(query)
        print(f"{english} -> {paiute}")
        print()


if __name__ == '__main__':
    main()