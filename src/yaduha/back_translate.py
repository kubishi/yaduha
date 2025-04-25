import json
import logging
from typing import Callable, Optional
from openai.types.chat import ChatCompletion

import dotenv

from .common import get_openai_client
from .sentence_builder import NOUNS, Object, Subject, Verb, POSSESSIVE_PRONOUNS

dotenv.load_dotenv()

client = get_openai_client()

def get_english_structure(subject_noun: str,
                          subject_noun_nominalizer: Optional[str],
                          subject_suffix: Optional[str],
                          subject_possessive_pronoun: Optional[str],
                          verb: Optional[str],
                          verb_tense: Optional[str],
                          object_pronoun: Optional[str],
                          object_noun: Optional[str],
                          object_noun_nominalizer: Optional[str],
                          object_suffix: Optional[str],
                          object_possessive_pronoun: Optional[str]) -> list:
    sentence_details = []

    subject_info = {'part_of_speech': 'subject'}
    if subject_noun_nominalizer is not None:
        subject_info['word'] = {**Verb.TRANSITIVE_VERBS, **Verb.INTRANSITIVE_VERBS}.get(subject_noun, subject_noun)
        subject_info['agent_nominalizer'] = Verb.NOMINALIZER_TENSES[subject_noun_nominalizer]
        subject_info['positional'] = Subject.SUFFIXES[subject_suffix]
    elif subject_noun in NOUNS:
        subject_info['word'] = NOUNS[subject_noun]
        subject_info['positional'] = Subject.SUFFIXES[subject_suffix]
    elif subject_noun in Subject.PRONOUNS:
        subject_info['word'] = Subject.PRONOUNS[subject_noun]
    else: # unknown word
        subject_info['word'] = subject_noun
    if subject_possessive_pronoun and subject_possessive_pronoun in POSSESSIVE_PRONOUNS:
        subject_info['possessive'] = POSSESSIVE_PRONOUNS[subject_possessive_pronoun]
    sentence_details.append(subject_info)
    
    object_info = {'part_of_speech': 'object'}
    if object_possessive_pronoun and object_possessive_pronoun in POSSESSIVE_PRONOUNS:
        object_info['possessive'] = POSSESSIVE_PRONOUNS[object_possessive_pronoun]
    
    plural_keywords = ['plural', 'you all', 'they', 'them', 'we', 'us']
    if object_pronoun and any(kw in Object.PRONOUNS[object_pronoun] for kw in plural_keywords):
        object_info['plural'] = True
    if object_noun_nominalizer is not None:
        object_info['word'] = {**Verb.TRANSITIVE_VERBS, **Verb.INTRANSITIVE_VERBS}.get(object_noun, object_noun)
        object_info['agent_nominalizer'] = Verb.NOMINALIZER_TENSES[object_noun_nominalizer]
        object_info['positional'] = Object.SUFFIXES[object_suffix]
        sentence_details.append(object_info)
    elif object_noun in NOUNS:
        object_info['word'] = NOUNS[object_noun]
        object_info['positional'] = Object.SUFFIXES[object_suffix]
        sentence_details.append(object_info)
    elif object_pronoun and not object_noun: # object pronoun
        object_info['word'] = Object.PRONOUNS[object_pronoun]
        sentence_details.append(object_info)
    elif (object_noun or '').strip(): # unknown word
        object_info['word'] = object_noun
        object_info['positional'] = Object.SUFFIXES[object_suffix]
        sentence_details.append(object_info)

    verb_info = {'part_of_speech': 'verb'}
    verb_info['word'] = Verb.TRANSITIVE_VERBS.get(verb, Verb.INTRANSITIVE_VERBS.get(verb))
    if verb_info['word'] is None:
        # raise Exception(f"Invalid verb: {verb}")
        verb_info['word'] = verb
    verb_info['tense'] = Verb.TENSES[verb_tense]
    sentence_details.append(verb_info)

    return sentence_details

def translate(subject_noun: str,
              subject_noun_nominalizer: Optional[str],
              subject_suffix: Optional[str],
              subject_possessive_pronoun: Optional[str],
              verb: Optional[str],
              verb_tense: Optional[str],
              object_pronoun: Optional[str],
              object_noun: Optional[str],
              object_noun_nominalizer: Optional[str],
              object_suffix: Optional[str],
              object_possessive_pronoun: Optional[str],
              model: str = 'gpt-4o',
              res_callback: Optional[Callable[[ChatCompletion], None]] = None) -> str:
    choices = dict(
        subject_noun=subject_noun,
        subject_noun_nominalizer=subject_noun_nominalizer,
        subject_suffix=subject_suffix,
        subject_possessive_pronoun=subject_possessive_pronoun,
        verb=verb,
        verb_tense=verb_tense,
        object_pronoun=object_pronoun,
        object_noun=object_noun,
        object_noun_nominalizer=object_noun_nominalizer,
        object_suffix=object_suffix,
        object_possessive_pronoun=object_possessive_pronoun
    )
    structure = get_english_structure(**choices)

    examples = [
        {
            'role': 'user', 
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'proximal', 'word': 'wood'},
                {'part_of_speech': 'object', 'positional': 'proximal', 'word': 'dog', 'possessive': 'my'},
                {'part_of_speech': 'verb', 'tense': 'present ongoing (-ing)', 'word': 'see'}]
            )
        },
        {'role': 'assistant', 'content': '(This wood) is seeing (my dog).'},
        {
            'role': 'user',
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'proximal', 'word': 'cup', 'possessive': 'my'},
                 {'part_of_speech': 'object', 'positional': 'distal', 'word': 'cup', 'plural': True},
                 {'part_of_speech': 'verb', 'tense': 'future (will)', 'word': 'eat'}]
            )
        },
        {'role': 'assistant', 'content': '(My cup) will eat (those cups).'},
        {
            'role': 'user',
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'distal', 'word': 'pinenuts', 'possessive': 'their own'},
                 {'part_of_speech': 'object', 'positional': 'distal', 'word': 'horse'},
                 {'part_of_speech': 'verb', 'tense': 'future (will)', 'word': 'see'}]
            )
        },
        {'role': 'assistant', 'content': '(Their own pinenuts) will see (that horse).'},
        # sawa-dü-ii kwati-deika ma-buni-ku 
        {
            'role': 'user',
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'proximal', 'word': 'cook', 'agent_nominalizer': 'present'},
                 {'part_of_speech': 'object', 'positional': 'proximal', 'word': 'hit', 'agent_nominalizer': 'present'},
                 {'part_of_speech': 'verb', 'tense': 'completive (past)', 'word': 'see'}]
            )
        },
        {'role': 'assistant', 'content': '(This one who cooks) saw (that one who hits).'},
        # # nia-pü-uu naka-wei-neika ma-dsibui-wei 
        {
            'role': 'user',
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'distal', 'word': 'read', 'agent_nominalizer': 'future', 'possessive': 'our'},
                 {'part_of_speech': 'object', 'positional': 'proximal', 'word': 'hear', 'agent_nominalizer': 'future'},
                 {'part_of_speech': 'verb', 'tense': 'future', 'word': 'climb'}]
            )
        },
        {'role': 'assistant', 'content': '(Our future reader) will climb (the one who will hear).'},
    ]
    logging.debug(json.dumps(structure, indent=2))
    messages = [
        {
            'role': 'system',
            'content': (
                'You are an assistant for translating structured sentences into natural English sentences.'
            )
        },
        *examples,
        {'role': 'user', 'content': json.dumps(structure)}
    ]
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=10,
        temperature=0.0
    )
    if res_callback:
        res_callback(res)
    translation = res.choices[-1].message.content
    translation = translation.replace('(', '').replace(')', '')
    return translation
