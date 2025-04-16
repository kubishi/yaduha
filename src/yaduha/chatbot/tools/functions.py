import requests
import pathlib
import dotenv
from openai import OpenAI
from tools.grammar import search_grammar as _search_grammar
import json
import os

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

thisdir = pathlib.Path(__file__).parent.absolute()

KUBISHI_API_URL = "https://dictionary.kubishi.com/api"

def search_english(query):
    response = requests.get(f"{KUBISHI_API_URL}/search/english", params={"query": query})
    response.raise_for_status()
    res_json = response.json()
    words = {word["lexical_unit"]: word["senses"][0].get("gloss") for word in res_json if word.get("senses")}
    return words

def search_paiute(query):
    response = requests.get(f"{KUBISHI_API_URL}/search/paiute", params={"query": query})
    response.raise_for_status()
    res_json = response.json()
    words = {word["lexical_unit"]: word["senses"][0].get("gloss") for word in res_json if word.get("senses")}
    return words

def search_grammar(query):
    return _search_grammar(
        query=query,
        limit=5
    )

def search_sentences(query):
    response = requests.get(f"{KUBISHI_API_URL}/search/sentence", params={"query": query})
    response.raise_for_status()
    res_json = response.json()

    top_sentences = []

    for sentence in res_json:
        top_sentences.append({sentence["translation"]: sentence["sentence"]})
    
    return top_sentences
