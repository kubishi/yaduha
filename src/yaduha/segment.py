"""Functions for segmenting complex sentences into sets of simple SVO or SV sentences."""
import functools
import hashlib
import logging
import os
import pathlib
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Union

from enum import Enum
from pydantic import BaseModel, ValidationError

import numpy as np
import openai
from openai.types.chat import ChatCompletion
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

cachedir = pathlib.Path.home() / '.cache' / 'yaduha'
cachedir.mkdir(exist_ok=True, parents=True)

def get_openai_client():
    try:
        return openai.Client(api_key=os.environ['OPENAI_API_KEY'])
    except KeyError:
        raise ValueError("OPENAI_API_KEY environment variable not set")

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

# Enumerations
class Proximity(str, Enum):
    proximal = "proximal"
    distal = "distal"

class Person(str, Enum):
    first = "first"
    second = "second"
    third = "third"

class Plurality(str, Enum):
    singular = "singular"
    dual = "dual"
    plural = "plural"

class Inclusivity(str, Enum):
    inclusive = "inclusive"
    exclusive = "exclusive"

class Tense(str, Enum):
    past = "past"
    present = "present"
    future = "future"

class Aspect(str, Enum):
    completive = "completive"
    continuous = "continuous"
    simple = "simple"
    perfect = "perfect"
    

class Verb(BaseModel):
    text: str
    lemma: str
    tense: Tense
    aspect: Aspect

# Models
class Pronoun(BaseModel):
    person: Person
    plurality: Plurality
    proximity: Proximity
    inclusivity: Inclusivity
    reflexive: bool

class Verb(BaseModel):
    lemma: str
    tense: Tense
    aspect: Aspect

class SubjectNoun(BaseModel):
    head: Union[str, Verb]
    possessive_determiner: Optional[Pronoun] = None
    proximity: Proximity
    plurality: Plurality

class ObjectNoun(BaseModel):
    head: Union[str, Verb]
    possessive_determiner: Optional[Pronoun] = None
    proximity: Proximity
    plurality: Plurality

class Sentence(BaseModel):
    subject: Union[SubjectNoun, Pronoun]
    verb: Verb
    object: Optional[Union[ObjectNoun, Pronoun]] = None

class SentenceList(BaseModel):
    sentences: List[Sentence]

# new split_sentence function using pydantic
try:
    EXAMPLE_SENTENCES = [
        {
            "sentence": "I am sitting in a chair.",
            "simple": "I am sitting.",
            "comparator": "I am sitting.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=Pronoun(
                            person=Person.first,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="sit",
                            tense=Tense.present,
                            aspect=Aspect.continuous
                        ),
                        object=None
                    )
                ]
            )
        },
        {
            "sentence": "The one who ran has seen Rebecca and met her.",
            "simple": "The one who ran has seen Rebecca. He met her.",
            "comparator": "The one who ran has seen [OBJECT]. He [VERB] her.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head=Verb(
                                lemma="run",
                                tense=Tense.past,
                                aspect=Aspect.completive
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular  
                        ),
                        verb=Verb(
                            lemma="see",
                            tense=Tense.present,
                            aspect=Aspect.perfect
                        ),
                        object=ObjectNoun(
                            head="Rebecca",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    ),
                    Sentence(
                        subject=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="meet",
                            tense=Tense.past,
                            aspect=Aspect.simple
                        ),
                        object=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=True
                        ),
                    )
                ]
            )
        },
        {
            "sentence": "The dogs were chasing their tails.",
            "simple": "The dogs were chasing their tails.",
            "comparator": "The dogs were chasing their tails.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head="dog",
                            proximity=Proximity.proximal,
                            plurality=Plurality.plural
                        ),
                        verb=Verb(
                            lemma="chase",
                            tense=Tense.past,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="tail",
                            possessive_determiner=Pronoun(
                                person=Person.third,
                                plurality=Plurality.plural,
                                proximity=Proximity.proximal,
                                inclusivity=Inclusivity.exclusive,
                                reflexive=True
                            ),
                            proximity=Proximity.proximal,
                            plurality=Plurality.plural
                        )
                    )
                ]
            )
        },
        {
            "sentence": "The fighter is eating his red apple.",
            "simple": "The fighter is eating his apple.",
            "comparator": "The [VERB]-er is eating his apple.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head=Verb(
                                lemma="fight",
                                tense=Tense.present,
                                aspect=Aspect.simple
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="eat",
                            tense=Tense.present,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="apple",
                            possessive_determiner=Pronoun(
                                person=Person.first,
                                plurality=Plurality.singular,
                                proximity=Proximity.proximal,
                                inclusivity=Inclusivity.exclusive,
                                reflexive=True
                            ),
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    )
                ]
            )
        },
        {
            "sentence": "The book sits on the table.",
            "simple": "The book sits.",
            "comparator": "The [SUBJECT] sits.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head="book",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="sit",
                            tense=Tense.present,
                            aspect=Aspect.simple
                        ),
                        object=None
                    )
                ]
            )
        },
        {
            "sentence": "The boy saw it.",
            "simple": "The boy saw it.",
            "comparator": "The [SUBJECT] saw it.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head="boy",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="see",
                            tense=Tense.past,
                            aspect=Aspect.completive
                        ),
                        object=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.distal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        )
                    )
                ]
            )
        },
        {
            "sentence": "I saw two men walking their dogs yesterday at Starbucks while drinking a cup of coffee",
            "simple": "I saw men. They were walking dogs. They were drinking coffee.",
            "comparator": "I saw [OBJECT]. They were walking dogs. They were drinking coffee.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=Pronoun(
                            person=Person.first,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="see",
                            tense=Tense.past,
                            aspect=Aspect.simple
                        ),
                        object=ObjectNoun(
                            head="man",
                            proximity=Proximity.distal,
                            plurality=Plurality.dual
                        )
                    ),
                    Sentence(
                        subject=Pronoun(
                            person=Person.third,
                            plurality=Plurality.dual,
                            proximity=Proximity.distal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="walk",
                            tense=Tense.past,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="dog",
                            possessive_determiner=Pronoun(
                                person=Person.third,
                                plurality=Plurality.dual,
                                proximity=Proximity.proximal,
                                inclusivity=Inclusivity.exclusive,
                                reflexive=True
                            ),
                            proximity=Proximity.proximal,
                            plurality=Plurality.plural
                        )
                    ),
                    Sentence(
                        subject=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="drink",
                            tense=Tense.past,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="coffee",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    )
                ]
            )
        },
        {
            "sentence": "That runner down the street will eat the one that has fallen.",
            "simple": "That runner will eat the one that has fallen.",
            "comparator": "That runner will eat the one that has fallen.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head=Verb(
                                lemma="run",
                                tense=Tense.present,
                                aspect=Aspect.simple
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.distal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="eat",
                            tense=Tense.future,
                            aspect=Aspect.simple
                        ),
                        object=ObjectNoun(
                            head=Verb(
                                lemma="fall",
                                tense=Tense.present,
                                aspect=Aspect.perfect
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    )
                ]
            )
        },
    ]
except ValidationError as exc:
    print(exc.errors())
    raise exc

def split_sentence(sentence: str, model: str, res_callback: Optional[Callable[[ChatCompletion], None]] = None) -> SentenceList:
    messages = [
        {
            "role": "system",
            "content": (
                'You are an assistant that splits user input sentences into a set of simple SVO or SV sentences. '
                'The set of simple sentences should be as semantically equivalent as possible to the user input sentence. '
                'No adjectives, adverbs, prepositions, or conjunctions should be added to the simple sentences. '
                'Indirect objects and objects of prepositions should not be included in the simple sentences. '
                'Subjects and objects can be verbs (via nominalization) '
                '(e.g., "run" -> "the runner", "the one who ran", "the one who will run"). '
            )
        }
    ]

    for example in EXAMPLE_SENTENCES:
        sentences_obj: SentenceList = example['response']
        messages.append({
            'role': 'user',
            'content': example['sentence']
        })
        messages.append({
            'role': 'assistant',
            'content': sentences_obj.model_dump_json(),
        })

    messages.append({
        'role': 'user',
        'content': sentence
    })

    client = get_openai_client()
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=SentenceList
    )

    if res_callback:
        res_callback(response)

    sentences = response.choices[0].message.parsed

    return sentences



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

