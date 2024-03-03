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

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
cache = FanoutCache(thisdir / ".cache", shards=64)

# oai_client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])


@functools.lru_cache(maxsize=1000)
def get_model(model: str) -> SentenceTransformer:
    return SentenceTransformer(model)


@functools.lru_cache(maxsize=1000)
def semantic_similarity_spacy(sentence1: str, sentence2: str) -> float:
    """Compute the semantic similarity between two sentences using spaCy.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The semantic similarity between the two sentences.
    """
    nlp = spacy.load("en_core_web_md")

    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    similarity = doc1.similarity(doc2)
    return similarity


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)

@functools.lru_cache(maxsize=1000)
def semantic_similarity_bert(sentence1: str, sentence2: str) -> float:
    with torch.no_grad():
        inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs1 = model(**inputs1)
        inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs2 = model(**inputs2)

        # Use the average of the last hidden states as sentence embeddings
        emb1 = outputs1.last_hidden_state.mean(dim=1)
        emb2 = outputs2.last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        return (similarity + 1) / 2  # Scale to 0-1 range

def semantic_similarity_sentence_transformers(sentence1: str, sentence2: str, model: str) -> float:
    embedder = get_model(model)
    emb1 = embedder.encode(sentence1, convert_to_tensor=True)
    emb2 = embedder.encode(sentence2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return (similarity + 1) / 2  # Scale to 0-1 range


def semantic_similarity_openai(sentence1: str, sentence2: str, model: str) -> float:
    embeddings = _get_openai_embeddings(model, sentence1, sentence2)
    # convert python lists to numpy arrays
    emb1 = np.array(embeddings[sentence1])
    emb2 = np.array(embeddings[sentence2])
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return (similarity + 1) / 2  # Scale to 0-1 range


def _get_openai_embeddings(model: str, *sentences: str) -> Dict[str, np.ndarray]:
    savedir = thisdir / ".results" / "embeddings" / model
    savedir.mkdir(exist_ok=True, parents=True)
    # load cached embeddings from disk
    embeddings = {}
    for sentence in sentences:
        sentence_id = hashlib.md5(sentence.encode()).hexdigest()
        try:
            with open(savedir / f"{sentence_id}.npy", "rb") as f:
                embeddings[sentence] = np.load(f)
        except FileNotFoundError:
            pass

    new_sentences = [s for s in sentences if s not in embeddings]
    if new_sentences:
        res = openai.embeddings.create(input=new_sentences, model=model, encoding_format="float")
        # save embeddings to disk
        for sentence, embedding in zip(new_sentences, res.data):
            emb = np.array(embedding.embedding)
            sentence_id = hashlib.md5(sentence.encode()).hexdigest()
            with open(savedir / f"{sentence_id}.npy", "wb") as f:
                np.save(f, emb)
            embeddings[sentence] = emb

    return embeddings