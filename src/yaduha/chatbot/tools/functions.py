from typing import Dict, List
import requests
import pathlib
import dotenv
from openai import OpenAI
import json
import os

from yaduha.chatbot.tools.grammar import search_grammar as _search_grammar

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

thisdir = pathlib.Path(__file__).parent.absolute()

KUBISHI_API_URL = "https://dictionary.kubishi.com/api"

def format_word(response_words: List) -> List:
    """
    Format the word response from the API into a string.
    """
    words = []
    for word in response_words:
        senses = []
        for sense in word["senses"]:
            sense_info = {}
            sense_info['gloss'] = sense.get("gloss")
            sense_info['definition'] = sense.get("definition")
            sense_info['examples'] = []
            if sense.get("examples"):
                for example in sense["examples"]:
                    sense_info['examples'].append({
                        "form": example["form"],
                        "translation": example["translation"]
                    })

            senses.append(sense_info)
        words.append({
            "lexical_unit": word["lexical_unit"],
            "senses": senses
        })
    
    return words

def search_english(query):
    response = requests.get(f"{KUBISHI_API_URL}/search/english", params={"query": query})
    response.raise_for_status()
    res_json: Dict = response.json()
    return format_word(res_json)
    # words = {word["lexical_unit"]: word["senses"][0].get("gloss") for word in res_json if word.get("senses")}
    # return words

def search_paiute(query):
    response = requests.get(f"{KUBISHI_API_URL}/search/paiute", params={"query": query})
    response.raise_for_status()
    res_json = response.json()
    return format_word(res_json)
    # words = {word["lexical_unit"]: word["senses"][0].get("gloss") for word in res_json if word.get("senses")}
    # return words

def search_grammar(query):
    return _search_grammar(
        query=query,
        limit=5
    )

def search_sentences(query):
    response = requests.get(f"{KUBISHI_API_URL}/search/sentence", params={"query": query})
    response.raise_for_status()
    res_json = response.json()
    infos = []
    for sentence in res_json:
        infos.append({
            "sentence": sentence["sentence"],
            "translation": sentence["translation"]
        })
    return infos

    # top_sentences = []

    # for sentence in res_json:
    #     top_sentences.append({sentence["translation"]: sentence["sentence"]})
    
    # return top_sentences
