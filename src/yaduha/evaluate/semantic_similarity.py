import functools
import hashlib
import logging
import pathlib
from typing import Dict, List, TYPE_CHECKING

import numpy as np
from yaduha.common import get_openai_client

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

cachedir = pathlib.Path.home() / '.cache' / 'yaduha'
cachedir.mkdir(exist_ok=True, parents=True)

nlp = None
@functools.lru_cache(maxsize=1000)
def semantic_similarity_spacy(sentence1: str, sentence2: str) -> float:
    """Compute the semantic similarity between two sentences using spaCy.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The semantic similarity between the two sentences.
    """
    global nlp
    import spacy
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            from spacy.cli import download
            logging.info("Downloading spaCy model 'en_core_web_md'")
            download('en_core_web_md')
            nlp = spacy.load("en_core_web_md")
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    similarity = doc1.similarity(doc2)
    return similarity

tokenizer = None
model = None
@functools.lru_cache(maxsize=1000)
def semantic_similarity_bert(sentence1: str, sentence2: str) -> float:
    global tokenizer, model
    import torch
    from transformers import BertModel, BertTokenizer

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if model is None:
        model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    with torch.no_grad():
        inputs1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs1 = model(**inputs1)
        inputs2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs2 = model(**inputs2)
        
        # Use the average of the last hidden states as sentence embeddings
        emb1 = outputs1.last_hidden_state.mean(dim=1)
        emb2 = outputs2.last_hidden_state.mean(dim=1)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        return (similarity + 1) / 2  # Scale to 0-1 range

@functools.lru_cache(maxsize=1000)
def get_model(model: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model)

def semantic_similarity_sentence_transformers(sentence1: str, sentence2: str, model: str) -> float:
    from sentence_transformers import util
    embedder = get_model(model)
    emb1 = embedder.encode(sentence1, convert_to_tensor=True)
    emb2 = embedder.encode(sentence2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return (similarity + 1) / 2  # Scale to 0-1 range

def semantic_similarity_sentence_transforms_all_combinations(sentences: List[str], model: str) -> np.ndarray:
    from sentence_transformers import util
    embedder = get_model(model)
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(embeddings, embeddings)
    similarities = similarities.cpu().numpy()
    # scale to 0-1 range
    similarities = (similarities + 1) / 2
    return similarities

tf_tokenizer = None
tf_model = None
def semantic_similarity_transformers_all_combinations(sentences: List[str], model: str) -> np.ndarray:
    global tf_tokenizer, tf_model
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Load model from HuggingFace Hub
    if tf_tokenizer is None:
        tf_tokenizer = AutoTokenizer.from_pretrained(model)
    if tf_model is None:
        tf_model = AutoModel.from_pretrained(model)

    # Tokenize sentences
    encoded_input = tf_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = tf_model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Compute cosine similarity
    similarities = F.cosine_similarity(sentence_embeddings.unsqueeze(1), sentence_embeddings.unsqueeze(0), dim=2)
    similarities = similarities.cpu().numpy()
    # scale to 0-1 range
    similarities = (similarities + 1) / 2
    return similarities

def semantic_similarity_transformers(sentence1: str, sentence2: str, model: str) -> float:
    # use semantic_similarity_transformers_all_combinations to compute similarity
    similarities = semantic_similarity_transformers_all_combinations([sentence1, sentence2], model)
    return similarities[0, 1]

def _get_openai_embeddings(model: str, *sentences: str) -> Dict[str, np.ndarray]:
    savedir = cachedir / '.results' / 'embeddings' / model
    savedir.mkdir(exist_ok=True, parents=True)
    # load cached embeddings from disk
    embeddings = {}
    for sentence in sentences:
        sentence_id = hashlib.md5(sentence.encode()).hexdigest()
        try:
            with open(savedir / f'{sentence_id}.npy', 'rb') as f:
                embeddings[sentence] = np.load(f)
        except FileNotFoundError:
            pass

    new_sentences = [s for s in sentences if s not in embeddings]
    if new_sentences:
        res = get_openai_client().embeddings.create(
            input=new_sentences,
            model=model,
            encoding_format="float"
        )
        # save embeddings to disk
        for sentence, embedding in zip(new_sentences, res.data):
            emb  = np.array(embedding.embedding)
            sentence_id = hashlib.md5(sentence.encode()).hexdigest()
            with open(savedir / f'{sentence_id}.npy', 'wb') as f:
                np.save(f, emb)
            embeddings[sentence] = emb

    return embeddings

def semantic_similarity_openai(sentence1: str, sentence2: str, model: str) -> float:
    from sentence_transformers import util
    embeddings = _get_openai_embeddings(model, sentence1, sentence2)
    # convert python lists to numpy arrays
    emb1 = np.array(embeddings[sentence1])
    emb2 = np.array(embeddings[sentence2])
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return (similarity + 1) / 2  # Scale to 0-1 range

def semantic_similarity_openai_all_combinations(sentences: List[str], model: str) -> np.ndarray:
    from sentence_transformers import util
    embeddings = _get_openai_embeddings(model, *sentences)
    # convert python lists to numpy arrays
    embeddings = np.array([embeddings[s] for s in sentences])
    similarities = util.pytorch_cos_sim(embeddings, embeddings)
    similarities = similarities.cpu().numpy()
    # scale to 0-1 range
    similarities = (similarities + 1) / 2
    return similarities