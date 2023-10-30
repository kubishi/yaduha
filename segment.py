"""Functions for segmenting complex sentences into sets of simple SVO or SV sentences."""
import json
import os
import pprint
from typing import Dict, List

import dotenv
import openai
import spacy

dotenv.load_dotenv()

# Load the medium English model with word vectors
nlp = spacy.load("en_core_web_md")

openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

def semantic_similarity_spacy(sentence1: str, sentence2: str) -> float:
    """Compute the semantic similarity between two sentences using spaCy.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The semantic similarity between the two sentences.
    """
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    similarity = doc1.similarity(doc2)
    return similarity

sentence_schema = {
  "type": "object",
  "properties": {
    "subject": {
      "type": "string",
      "description": "The subject of the sentence. Must be a single word."
    },
    "verb": {
      "type": "string",
      "description": "The present-tense verb of the sentence. Must be a single word (infinitive without 'to')."
    },
    "verb_tense": {
        "type": "string",
        "description": "The tense of the verb. Must be one of: past, present, future.",
        "enum": ["past", "present", "future", "past_continuous", "present_continuous"]
    },
    "object": {
      "type": "string",
      "description": "The object of the sentence (optional). Must be a single word."
    }
  },
  "required": ["subject", "verb", "verb_tense"]
}

def split_sentence(sentence: str) -> List[Dict[str, str]]:
    """Split a sentence into a set of simple SVO or SV sentences.

    Args:
        sentence (str): The sentence to split.

    Returns:
        list: A list of simple sentences.
    """
    functions = [
        {
            'name': 'set_sentences',
            'description': 'Set the simple sentences.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sentences': sentence_schema
                },
                'required': ['sentences']
            }
        }
    ]

    messages = [
        {'role': 'system', 'content': "".join([
            'You are an assistant that splits user input sentences into a set of simple SVO or SV sentences. ',
            'The set of simple sentences should be as semantically equivalent as possible to the user input sentence. ',
            'No adjectives, adverbs, prepositions, or conjunctions should be added to the simple sentences. ',
            'Indirect objects and objects of prepositions should not be included in the simple sentences. ',
        ])},
        {'role': 'user', 'content': 'I am sitting in a chair.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'I', 'verb': 'sit', 'verb_tense': 'present_continuous', 'object': None},
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': 'I saw a man walking his dog yesterday at Starbucks while drinking a cup of coffee'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'I', 'verb': 'see', 'verb_tense': 'past', 'object': 'man'},
                        {'subject': 'man', 'verb': 'walk', 'verb_tense': 'past_continuous', 'object': 'dog'},
                        {'subject': 'man', 'verb': 'drink', 'verb_tense': 'past_continuous', 'object': 'coffee'}
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': sentence},
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        functions=functions,
        function_call={'name': 'set_sentences'},
        temperature=0.0,
    )
    response_message = response["choices"][0]["message"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    return function_args.get('sentences')

def make_sentence(sentence: Dict) -> str:
    """Generate a simple SVO or SV sentence from a schema.

    Args:
        sentence (dict): The sentence schema.

    Returns:
        str: The generated sentence.
    """
    functions = [
        {
            'name': 'make_sentence',
            'description': 'Write a simple natural language sentence.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sentence': {'type': 'string'}
                },
                'required': ['sentence']
            }
        }
    ]
    messages = [
        {
            'role': 'system',
            'content': 'You are an assistant takes structured data and generates simple SVO or SV natural language sentence. Only add add necessary articles and conjugations. Do not add any other words.'
        },
        {
            'role': 'system',
            'content': "{'subject': 'I', 'verb': 'see', 'verb_tense': 'past', 'object': 'man'}"
        },
        {
            'role': 'assistant',
            'content': None,
            'function_call': {
                'arguments': json.dumps({'sentence': 'I saw a man'}),
                'name': 'make_sentence'
            }
        },
        {
            'role': 'user',
            'content': json.dumps(sentence)
        }
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        functions=functions,
        function_call={'name': 'make_sentence'},
        temperature=0.0,
    )
    response_message = response["choices"][0]["message"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    return function_args.get('sentence')


def main(): # pylint: disable=missing-function-docstring
    source_sentences = [
        "The dog fell.",
        "The dog fell yesterday.",
        "The dog was running yesterday and fell.",
        "The dog was running yesterday and fell while chasing a cat.",
        "The dog sat in the house.",
        "I gave him bread.",
        "The dog and the cat were running."
    ]
    for source_sentence in source_sentences:
        simple_sentences = split_sentence(source_sentence)
        print(simple_sentences)
        simple_nl_sentence = '. '.join([make_sentence(sentence) for sentence in simple_sentences]) + '.'

        print(f"Source sentence: {source_sentence}")
        print(f"Simple sentences: {simple_nl_sentence}")
        similarity = semantic_similarity_spacy(source_sentence, simple_nl_sentence)
        print(f"Semantic similarity: {similarity:0.3f}")
        print()

if __name__ == '__main__':
    main()
