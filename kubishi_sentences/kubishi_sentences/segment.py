"""Functions for segmenting complex sentences into sets of simple SVO or SV sentences."""

import functools
import hashlib
import json
import os
import pathlib
import pprint
from typing import Callable, Dict, List, Optional

import dotenv
import numpy as np
import openai
import pandas as pd
import rbo
import spacy
import torch
from diskcache import FanoutCache
from openai.types.chat import ChatCompletion
from sentence_transformers import SentenceTransformer, util
from transformers import BertModel, BertTokenizer

from kubishi_sentences.semantic_models import semantic_similarity_openai, semantic_similarity_bert, semantic_similarity_sentence_transformers, semantic_similarity_spacy

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
cache = FanoutCache(thisdir / ".cache", shards=64)

SENTENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "The subject of the sentence. Must be a single word and singular (not plural).",
        },
        "verb": {
            "type": "string",
            "description": "The present-tense verb of the sentence. Must be a single word (infinitive without 'to').",
        },
        "verb_tense": {
            "type": "string",
            "description": "The tense of the verb. Must be one of: past, present, future.",
            "enum": ["past", "present", "future", "past_continuous", "present_continuous"],
        },
        "object": {
            "type": "string",
            "description": "The object of the sentence (optional). Must be a single word and singular (not plural).",
        },
    },
    "required": ["subject", "verb", "verb_tense"],
}


def hash_dict(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    class HDict(dict):
        def __hash__(self):
            return hash(frozenset(self.items()))

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([HDict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: HDict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapped

def avg_displacement(truth: np.ndarray, arr: np.ndarray) -> float:
    """Compute the average displacement between two arrays.

    Computes the distance of each element to its proper position in the truth array
    and returns the average of these distances.
    """
    return np.mean(np.abs(np.argsort(truth) - np.argsort(arr)))



# @functools.lru_cache(maxsize=1000)
def split_sentence(
    sentence: str, model: str = None, res_callback: Optional[Callable[[ChatCompletion], None]] = None
) -> List[Dict]:
    """Split a sentence into a set of simple SVO or SV sentences.

    Args:
        sentence (str): The sentence to split.
        res_callback (Optional[Callable[[ChatCompletion], None]]): Callback function to be called with the completion response.

    Returns:
        list: A list of simple sentences.
    """
    if model is None:
        model = os.environ["OPENAI_MODEL"]
    functions = [
        {
            "name": "set_sentences",
            "description": "Set the simple sentences.",
            "parameters": {"type": "object", "properties": {"sentences": SENTENCE_SCHEMA}, "required": ["sentences"]},
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "".join(
                [
                    "You are an assistant that splits user input sentences into a set of simple SVO or SV sentences. ",
                    "The set of simple sentences should be as semantically equivalent as possible to the user input sentence. ",
                    "No adjectives, adverbs, prepositions, or conjunctions should be added to the simple sentences. ",
                    "Indirect objects and objects of prepositions should not be included in the simple sentences. ",
                ]
            ),
        },
        {"role": "user", "content": "I am sitting in a chair."},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps(
                    {
                        "sentences": [
                            {"subject": "I", "verb": "sit", "verb_tense": "present_continuous", "object": None},
                        ]
                    }
                ),
                "name": "set_sentences",
            },
        },
        {"role": "user", "content": "The dogs were chasing their tails."},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps(
                    {
                        "sentences": [
                            {"subject": "dog", "verb": "chase", "verb_tense": "past_continuous", "object": "tail"},
                        ]
                    }
                ),
                "name": "set_sentences",
            },
        },
        {
            "role": "user",
            "content": "I saw two men walking their dogs yesterday at Starbucks while drinking a cup of coffee",
        },
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps(
                    {
                        "sentences": [
                            {"subject": "I", "verb": "see", "verb_tense": "past", "object": "man"},
                            {"subject": "man", "verb": "walk", "verb_tense": "past_continuous", "object": "dog"},
                            {"subject": "man", "verb": "drink", "verb_tense": "past_continuous", "object": "coffee"},
                        ]
                    }
                ),
                "name": "set_sentences",
            },
        },
        {"role": "user", "content": sentence},
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call={"name": "set_sentences"},
        temperature=0.0,
        timeout=10,
    )

    if res_callback:
        res_callback(response)
    response_message = response.choices[0].message
    function_args = json.loads(response_message.function_call.arguments)
    return function_args.get("sentences")


@hash_dict
# @functools.lru_cache(maxsize=1000)
def make_sentence(
    sentence: Dict, model: str = None, res_callback: Optional[Callable[[ChatCompletion], None]] = None
) -> str:
    """Generate a simple SVO or SV sentence from a schema.

    Args:
        sentence (dict): The sentence schema.

    Returns:
        str: The generated sentence.
    """
    if model is None:
        model = os.environ["OPENAI_MODEL"]

    functions = [
        {
            "name": "make_sentence",
            "description": "Write a simple natural language sentence.",
            "parameters": {"type": "object", "properties": {"sentence": {"type": "string"}}, "required": ["sentence"]},
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are an assistant takes structured data and generates simple SVO or SV natural language sentence. Only add add necessary articles and conjugations. Do not add any other words.",
        },
        {"role": "system", "content": "{'subject': 'I', 'verb': 'see', 'verb_tense': 'past', 'object': 'man'}"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"arguments": json.dumps({"sentence": "I saw a man"}), "name": "make_sentence"},
        },
        {"role": "user", "content": json.dumps(sentence)},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call={"name": "make_sentence"},
        temperature=0.0,
        timeout=10,
    )
    if res_callback:
        res_callback(response)
    response_message = response.choices[0].message
    function_args = json.loads(response_message.function_call.arguments)
    return function_args.get("sentence")


def calculate_semantic_similarity_metrics_helper(base_sentence, comparison_sentences, similarity_func, ranking_algorithm):
    """
    Calculate semantic similarity and ranking for a base sentence and comparison sentences.

    This function calculates the semantic similarity between a base sentence and a list 
    of comparison sentences using a provided semantic similarity function. It also computes 
    the ranking similarity using the given ranking algorithm.

    Args:
        base_sentence (str): The original sentence against which comparison is made.
        comparison_sentences (list of str): List of sentences created from the base_sentence for comparison.
        similarity_func (callable): A function that computes semantic similarity between two sentences.
        ranking_algorithm (callable): A function or object capable of computing ranking similarity between sentences.

    Returns:
        tuple: A tuple containing two elements:
            - avg_displacement (float): The average displacement between sorted and original indices of similarities.
            - rbo_similarity (float): The Rank Biased Overlap similarity between the sorted and original lists of sentences.
    """

    similarities = np.array([similarity_func(base_sentence, s) for s in comparison_sentences])
    dist = np.mean(np.abs(np.argsort(-similarities) - np.arange(len(similarities))))

    sorted_sentences = [comparison_sentences[i] for i in np.argsort(-similarities)]
    
    rbo_similarity = ranking_algorithm(sorted_sentences, comparison_sentences).rbo()

    return dist, rbo_similarity

def calculate_similarity_metrics(sentences, similarity_funcs, ranking_algorithm = rbo.RankingSimilarity):
    """
    Calculate semantic similarity metrics for given sentences using various similarity functions.

    This function calculates semantic similarity metrics, including average displacement and RBO similarity, 
    for a set of sentences using multiple semantic similarity functions.

    Args:
        sentences (list of dict): List of dictionaries containing 'base' and 'sentences' keys,
            where 'base' represents the original sentence, and 'sentences' represents a list of
            sentences created from the base sentence for comparison.
        similarity_funcs (dict): A dictionary where keys are string identifiers for similarity functions
            and values are the corresponding similarity function objects.
        ranking_algorithm (callable): A function or object capable of computing ranking similarity between sentences.
            Default is `rbo.RankingSimilarity`.

    Returns:
        list of dict: A list of dictionaries containing calculated semantic similarity metrics for each sentence
            using various similarity functions. Each dictionary has keys 'sentence', 'similarity_func',
            'avg_displacement', and 'rbo'.
    """

    rows = []
    
    for sentence in sentences:
        base_sentence = sentence["base"]
        sentences = sentence["sentences"]

        for similarity_func_name, similarity_func in similarity_funcs.items():

            ranking_similarity, avg_displacement = calculate_semantic_similarity_metrics_helper(base_sentence, sentences, similarity_func, ranking_algorithm)

            row = {
                'sentence': base_sentence,
                'similarity_func': similarity_func_name,
                'avg_displacement': avg_displacement,
                'rbo': ranking_similarity,
            }

            rows.append(row)
   
    return rows


def main():
    sentences = json.loads((thisdir / "data" / "semantic_sentences.json").read_text())
    similarity_funcs = {
        "spacy": semantic_similarity_spacy,
        "bert": semantic_similarity_bert,
        "all-MiniLM-L6-v2": functools.partial(semantic_similarity_sentence_transformers, model="all-MiniLM-L6-v2"),
        "paraphrase-MiniLM-L6-v2": functools.partial(
            semantic_similarity_sentence_transformers, model="paraphrase-MiniLM-L6-v2"
        ),
        # "SFR-Embedding-Mistral": functools.partial(semantic_similarity_sentence_transformers, model='Salesforce/SFR-Embedding-Mistral'),
        "text-embedding-3-large": functools.partial(semantic_similarity_openai, model="text-embedding-3-large"),
        "text-embedding-3-small": functools.partial(semantic_similarity_openai, model="text-embedding-3-small"),
        "text-embedding-ada-002": functools.partial(semantic_similarity_openai, model="text-embedding-ada-002"),
    }

    rows = calculate_similarity_metrics(sentences, similarity_funcs)

    df = pd.DataFrame(rows)
    print(df)

    # compute stats for each similarity function
    stats = df.groupby("similarity_func").agg({"avg_displacement": ["mean", "std"], "rbo": ["mean", "std"]})
    stats = stats.sort_values(by=("rbo", "mean"), ascending=False)
    print(stats.round(3))
    print(stats.to_latex(float_format="%.3f", bold_rows=True, column_format="lcccc"))


if __name__ == "__main__":
    main()
