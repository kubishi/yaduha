"""Functions for translating simple sentences from English to Paiute."""
from copy import deepcopy
from functools import partial
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from openai.types.chat import ChatCompletion

from yaduha.translate.pipeline_sentence_builder import NOUNS, Object, Subject, Verb
from yaduha.translate.examples import EXAMPLE_SENTENCES
from yaduha.common import get_openai_client
from yaduha.translate.pipeline_syntax import (
    Sentence, SubjectNoun, Proximity, Person, Plurality,
    Inclusivity, Tense, Aspect,
    Pronoun, ObjectNoun, SentenceList
)
from yaduha.translate.pipeline_syntax import Verb as SegmentVerb
from yaduha.evaluate.semantic_similarity import (
    semantic_similarity_transformers,
    semantic_similarity_openai
)
from yaduha.translate.pipeline_back_translate import translate as translate_ovp_to_english
from yaduha.translate.base import Translator, Translation

SS_MODE = os.getenv('SS_MODE', 'sentence-transformers')

if SS_MODE == 'openai':
    semantic_similarity = partial(semantic_similarity_openai, model='text-embedding-ada-002')
else:
    semantic_similarity = partial(semantic_similarity_transformers, model='sentence-transformers/all-MiniLM-L6-v2')

R_TRANSITIVE_VERBS = {v: k for k, v in Verb.TRANSITIVE_VERBS.items()}
R_INTRANSITIVE_VERBS = {v: k for k, v in Verb.INTRANSITIVE_VERBS.items()}
R_NOUNS = {v: k for k, v in NOUNS.items()}

R_VERB_TENSES = {
    Tense.present: {
        Aspect.continuous: "ti",
        Aspect.completive: "ti",
        Aspect.simple: "dü",
        Aspect.perfect: "pü"
    },
    Tense.past: {
        Aspect.continuous: "ti",
        Aspect.completive: "ku",
        Aspect.simple: "ti",
        Aspect.perfect: "pü"
    },
    Tense.future: {
        Aspect.continuous: "wei",
        Aspect.completive: "wei",
        Aspect.simple: "wei",
        Aspect.perfect: "wei"
    },
}

OBJECT_PRONOUN_MAP = {
    'i': {
        'person': [Person.first],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': None,
        'reflexive': None,
    },
    'ü': {
        'person': [Person.second],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': None,
        'reflexive': None,
    },
    'u': {
        'person': [Person.third],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': [Proximity.distal],
        'reflexive': [False],
    },
    'a': {
        'person': [Person.third],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': [Proximity.proximal],
        'reflexive': [False],
    },
    'ma': {
        'person': [Person.third],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': [Proximity.proximal],
        'reflexive': [False],
    },
    'ui': {
        'person': [Person.third],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': [Proximity.distal],
        'reflexive': [False],
    },
    'ai': {
        'person': [Person.third],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': [Proximity.proximal],
        'reflexive': [False],
    },
    'mai': {
        'person': [Person.third],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': [Proximity.proximal],
        'reflexive': [False],
    },
    'ni': {
        'person': [Person.first],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': [Inclusivity.exclusive],
        'proximity': None,
        'reflexive': None,
    },
    'tei': {
        'person': [Person.first],
        'plurality': [Plurality.plural],
        'inclusivity':[ Inclusivity.inclusive],
        'proximity': None,
        'reflexive': None,
    },
    'ta': {
        'person': [Person.first],
        'plurality': [Plurality.dual],
        'inclusivity': [Inclusivity.inclusive],
        'proximity': None,
        'reflexive': None,
    },
    'üi': {
        'person': [Person.second],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': None,
        'reflexive': None,
    },
    'tü': {
        'person': [Person.third],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': None,
        'reflexive': [True],
    },
    'tüi': {
        'person': [Person.third],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': None,
        'reflexive': [True],
    },
}


SUBJECT_PRONOUN_MAP = {
    'nüü': {
        'person': [Person.first],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': None,
        'reflexive': None,
    },
    'üü': {
        'person': [Person.second],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': None,
        'reflexive': None,
    },
    'uhu': {
        'person': [Person.third],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': [Proximity.distal],
        'reflexive': None,
    },
    'mahu': {
        'person': [Person.third],
        'plurality': [Plurality.singular],
        'inclusivity': None,
        'proximity': [Proximity.proximal],
        'reflexive': None,
    },
    'uhuw̃a': {
        'person': [Person.third],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': [Proximity.distal],
        'reflexive': None,
    },
    'mahuw̃a': {
        'person': [Person.third],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': [Proximity.proximal],
        'reflexive': None,
    },
    'nüügwa': {
        'person': [Person.first],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': [Inclusivity.exclusive],
        'proximity': None,
        'reflexive': None,
    },
    'taagwa': {
        'person': [Person.first],
        'plurality': [Plurality.plural],
        'inclusivity': [Inclusivity.inclusive],
        'proximity': None,
        'reflexive': None,
    },
    'taa': {
        'person': [Person.first],
        'plurality': [Plurality.dual],
        'inclusivity': [Inclusivity.inclusive],
        'proximity': None,
        'reflexive': None,
    },
    'üügwa': {
        'person': [Person.second],
        'plurality': [Plurality.plural, Plurality.dual],
        'inclusivity': None,
        'proximity': None,
        'reflexive': None,
    }
}

POSSESSIVE_PRONOUN_MAP = OBJECT_PRONOUN_MAP.copy()

def _map_pronoun_to_string(pronoun: Pronoun, mapping: Dict[str, str]) -> str:
    """Map an object pronoun to its Paiute string representation."""
    matches = []
    for paiute, attributes in mapping.items():
        does_match = all(
            attributes[attr] is None or (getattr(pronoun, attr) in attributes[attr])
            for attr in ['person', 'plurality', 'inclusivity', 'proximity', 'reflexive']
        )
        if does_match:
            matches.append(paiute)

    if not matches:
        print(f"WARNING: No match found for {pronoun} in {mapping}")
    return random.choice(matches)

def _map_subject_pronoun_to_string(subject_pronoun: Pronoun) -> str:
    return _map_pronoun_to_string(subject_pronoun, SUBJECT_PRONOUN_MAP)

def _map_object_pronoun_to_string(object_pronoun: Pronoun) -> str:
    return _map_pronoun_to_string(object_pronoun, OBJECT_PRONOUN_MAP)

def _map_possessive_pronoun_to_string(possessive_pronoun: Pronoun) -> str:
    return _map_pronoun_to_string(possessive_pronoun, POSSESSIVE_PRONOUN_MAP)

def _map_proximity_to_subject_suffix(proximity: Proximity) -> str:
    return {'proximal': 'ii', 'distal': 'uu'}.get(proximity, 'ii')

def _map_proximity_to_object_suffix(proximity: Proximity) -> str:
    return {'proximal': 'eika', 'distal': 'oka'}.get(proximity, 'eika')

def _map_tense_to_suffix(tense: str, aspect: str) -> str:
    return R_VERB_TENSES.get(tense, {}).get(aspect, "dü")

def _map_tense_to_nominalizer(tense: Tense) -> str:
    return {
        Tense.present: 'dü',
        Tense.future: 'weidü',
    }.get(tense, "dü")

def translate_simple(sentence: Sentence) -> Tuple[Subject, Verb, Optional[Object]]:
    """Translate a structured English sentence (Sentence object) to Paiute."""

    # === SUBJECT ===
    subj = sentence.subject
    if isinstance(subj, Pronoun):
        pronoun_form = _map_subject_pronoun_to_string(subj)
        subject = Subject(pronoun_form, subject_noun_nominalizer=None, subject_suffix=None)
    elif isinstance(subj, SubjectNoun):
        head = subj.head
        if isinstance(head, str):
            noun = R_NOUNS.get(head.lower(), f"[{head}]")
            nominalizer = None
        elif isinstance(head, SegmentVerb):
            eng_lemma = head.lemma.lower().strip()
            noun = (
                R_INTRANSITIVE_VERBS.get(eng_lemma) or
                R_TRANSITIVE_VERBS.get(eng_lemma) or
                f"[{eng_lemma}]"
            )
            nominalizer = _map_tense_to_nominalizer(head.tense)
        else:
            raise ValueError("Unsupported subject head type")

        subj_possessive_pronoun = None
        if subj.possessive_determiner:
            subj_possessive_pronoun = _map_possessive_pronoun_to_string(subj.possessive_determiner)

        suffix = _map_proximity_to_subject_suffix(subj.proximity)
        subject = Subject(
            noun,
            subject_noun_nominalizer=nominalizer,
            subject_suffix=suffix,
            possessive_pronoun=subj_possessive_pronoun
        )
    else:
        raise ValueError("Invalid subject structure")

    # === VERB ===
    verb_input = sentence.verb
    eng_verb = verb_input.lemma.lower().strip()
    paiute_stem = R_TRANSITIVE_VERBS.get(eng_verb)
    if sentence.object is None:
        paiute_stem = R_INTRANSITIVE_VERBS.get(eng_verb, paiute_stem)

    if paiute_stem is None:
        paiute_stem = f"[{eng_verb}]"

    tense = _map_tense_to_suffix(verb_input.tense, verb_input.aspect)
    object_pronoun_prefix = None
    _object = None

    # === OBJECT ===
    if sentence.object:
        obj = sentence.object
        if isinstance(obj, Pronoun):
            object_pronoun_prefix = _map_object_pronoun_to_string(obj)
            matching_suffix = Object.get_matching_suffix(object_pronoun_prefix)
            verb = Verb(paiute_stem, tense, object_pronoun_prefix=object_pronoun_prefix)
            _object = None
        elif isinstance(obj, ObjectNoun):
            head = obj.head
            if isinstance(head, str):
                noun = R_NOUNS.get(head.lower(), f"[{head}]")
                nominalizer = None
            elif isinstance(head, SegmentVerb):
                eng_lemma = head.lemma.lower().strip()
                noun = (
                    R_TRANSITIVE_VERBS.get(eng_lemma) or
                    R_INTRANSITIVE_VERBS.get(eng_lemma) or
                    f"[{eng_lemma}]"
                )
                nominalizer = _map_tense_to_nominalizer(head.tense.value)
            else:
                raise ValueError("Unsupported object head type")
            
            obj_possessive_pronoun = None
            if obj.possessive_determiner:
                obj_possessive_pronoun = _map_possessive_pronoun_to_string(obj.possessive_determiner)

            suffix = _map_proximity_to_object_suffix(obj.proximity)
            _object = Object(
                noun,
                object_noun_nominalizer=nominalizer,
                object_suffix=suffix,
                possessive_pronoun=obj_possessive_pronoun
            )

            object_pronoun_prefix = _map_object_pronoun_to_string(
                Pronoun(
                    person=Person.third,
                    plurality=obj.plurality,
                    proximity=obj.proximity,
                    inclusivity=Inclusivity.exclusive,
                    reflexive=False,
                )
            )
            verb = Verb(paiute_stem, tense, object_pronoun_prefix=object_pronoun_prefix)
        else:
            raise ValueError("Invalid object structure")
    else:
        verb = Verb(paiute_stem, tense, object_pronoun_prefix=None)

    return subject, verb, _object


def order_sentence(subject: Subject, verb: Verb, _object: Optional[Object] = None) -> List[Union[Subject, Verb, Object]]:
    if subject.noun in Subject.PRONOUNS:
        sentence = [subject, verb] if _object is None else [subject, _object, verb]
    else:
        sentence = [verb, subject] if _object is None else [_object, subject, verb]
    return sentence

def comparator_sentence(simple_sentence: Sentence) -> Sentence:
    simple_sentence = deepcopy(simple_sentence)

    if isinstance(simple_sentence.subject, SubjectNoun):
        head = simple_sentence.subject.head
        if isinstance(head, str):
            if head not in R_NOUNS:
                simple_sentence.subject.head = "[SUBJECT]"
        elif isinstance(head, SegmentVerb):
            if head.lemma not in {*R_TRANSITIVE_VERBS, *R_INTRANSITIVE_VERBS}:
                simple_sentence.subject.head.lemma = "[VERB]"

    if isinstance(simple_sentence.verb, SegmentVerb):
        if simple_sentence.verb.lemma not in {*R_TRANSITIVE_VERBS, *R_INTRANSITIVE_VERBS}:
            simple_sentence.verb.lemma = "[VERB]"
    
    if simple_sentence.object:
        if isinstance(simple_sentence.object, ObjectNoun):
            head = simple_sentence.object.head
            if isinstance(head, str):
                if head not in R_NOUNS:
                    simple_sentence.object.head = "[OBJECT]"
            elif isinstance(head, SegmentVerb):
                if head.lemma not in {*R_TRANSITIVE_VERBS, *R_INTRANSITIVE_VERBS}:
                    simple_sentence.object.head.lemma = "[VERB]"

    return simple_sentence


# create new make_sentence function using pydantic
def make_sentence(sentence: Sentence, model: str, res_callback: Optional[Callable[[ChatCompletion], None]] = None) -> str:
    messages = [
        {
            'role': 'system',
            'content': (
                'You are an assistant that takes structured data and generates simple SVO or SV natural language sentence. '
                'Only add necessary articles and conjugations. '
                'Do not add any other words.'
                'When a subject or object is a verb, they are "nominalized" verbs as "past", "present", or "future" '
                '(e.g., "run" -> "the runner", "the one who ran", "the one who will run"; "drink" -> "the drinker", "the one who drank", "the one who will drink"). '
                'Leave words wrapped in square brackets (e.g. [SUBJECT]) as they are. '
            )
        }
    ]

    for example in EXAMPLE_SENTENCES:
        sentences_obj: SentenceList = example['response']
        comparator_sentences = SentenceList(
            sentences=[
                comparator_sentence(sentence) for sentence in sentences_obj.sentences
            ]
        )
        messages.append({
            'role': 'user',
            'content': comparator_sentences.model_dump_json()
        })
        messages.append({
            'role': 'assistant',
            'content': example['comparator']
        })

    messages.append({
        'role': 'user',
        'content': sentence.model_dump_json()
    })

    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    if res_callback:
        res_callback(response)

    return response.choices[0].message.content

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
    print(sentences.model_dump_json())
    return sentences

class PipelineTranslator(Translator):
    def __init__(self, model: str):
        self.model = model

    def translate(self, sentence: str) -> Translation:
        start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        model_calls = 0
        def res_callback(res: ChatCompletion):
            nonlocal prompt_tokens, completion_tokens, model_calls
            prompt_tokens += res.usage.prompt_tokens
            completion_tokens += res.usage.completion_tokens
            model_calls += 1

        prompt_tokens_back = 0
        completion_tokens_back = 0
        model_calls_back = 0
        def res_callback_backwards(res: ChatCompletion):
            nonlocal prompt_tokens_back, completion_tokens_back, model_calls_back
            prompt_tokens_back += res.usage.prompt_tokens
            completion_tokens_back += res.usage.completion_tokens
            model_calls_back += 1

        simple_sentences = split_sentence(sentence, model=self.model, res_callback=res_callback)
        comparator_sentences = []
        target_simple_sentences = []
        backwards_translations = []
        back_translation_time = 0
        for simple_sentence in simple_sentences.sentences:
            comparator_sentences.append(comparator_sentence(simple_sentence))
            subject, verb, _object = translate_simple(simple_sentence)
            target_simple_sentence = order_sentence(subject, verb, _object)
            target_simple_sentences.append(" ".join(map(str, target_simple_sentence)))
            back_translation_start_time = time.time()
            backwards_translations.append(
                translate_ovp_to_english(
                    subject_noun=subject.noun,
                    subject_noun_nominalizer=subject.subject_noun_nominalizer,
                    subject_suffix=subject.subject_suffix,
                    subject_possessive_pronoun=subject.possessive_pronoun,
                    verb=verb.verb_stem,
                    verb_tense=verb.tense_suffix,
                    object_pronoun=verb.object_pronoun_prefix,
                    object_noun=_object.noun if _object else None,
                    object_noun_nominalizer=_object.object_noun_nominalizer if _object else None,
                    object_suffix=_object.object_suffix if _object else None,
                    object_possessive_pronoun=_object.possessive_pronoun if _object else None,
                    res_callback=res_callback_backwards
                ).strip(".")
            )
            back_translation_time += time.time() - back_translation_start_time

        # simple_sentences_nl = ". ".join([make_sentence(sentence, model=self.model, res_callback=res_callback) for sentence in simple_sentences]) + '.'
        simple_sentences_nl = make_sentence(simple_sentences, model=self.model, res_callback=res_callback)
        comparator_sentence_nl = make_sentence(SentenceList(sentences=comparator_sentences), model=self.model, res_callback=res_callback_backwards)
        target_simple_sentence_nl = ". ".join(target_simple_sentences) + '.'
        backwards_translation_nl = ". ".join(backwards_translations) + '.'

        translation_time = (time.time() - start_time) - back_translation_time
        return Translation(
            source=sentence,
            target=target_simple_sentence_nl,
            back_translation=backwards_translation_nl,
            translation_prompt_tokens=prompt_tokens,
            translation_completion_tokens=completion_tokens,
            translation_time=translation_time,
            back_translation_prompt_tokens=prompt_tokens_back,
            back_translation_completion_tokens=completion_tokens_back,
            back_translation_time=back_translation_time,
            metadata={
                'simple': simple_sentences_nl,
                'comparator': comparator_sentence_nl,
                'model_calls': model_calls,
                'back_model_calls': model_calls_back,
            }
        )
