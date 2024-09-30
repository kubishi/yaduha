"""Functions for translating simple sentences from English to Paiute."""
import argparse
from functools import partial
import json
import logging
import os
import pathlib
import random
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dotenv
import numpy as np
from openai.types.chat import ChatCompletion
import pandas as pd

from sentence_builder import NOUNS, Object, Subject, Verb
from segment import make_sentence, split_sentence
from segment import semantic_similarity_transformers, semantic_similarity_openai
from translate_ovp2eng import translate as translate_ovp_to_english

dotenv.load_dotenv()

SS_MODE = os.getenv('SS_MODE', 'sentence-transformers')

if SS_MODE == 'openai':
    semantic_similarity = partial(semantic_similarity_openai, model='text-embedding-ada-002')
else:
    semantic_similarity = partial(semantic_similarity_transformers, model='sentence-transformers/all-MiniLM-L6-v2')

thisdir = pathlib.Path(__file__).parent.absolute()

R_TRANSITIVE_VERBS = {v: k for k, v in Verb.TRANSITIVE_VERBS.items()}
R_INTRANSITIVE_VERBS = {v: k for k, v in Verb.INTRANSITIVE_VERBS.items()}
R_NOUNS = {v: k for k, v in NOUNS.items()}

R_OBJECT_PRONOUNS = {
    'me': ['i'],
    'you': ['ü'],
    'him': ['u', 'a', 'ma'],
    'her': ['u', 'a', 'ma'],
    'it': ['u', 'a', 'ma'],
    'us': ['tei', 'ni'],
    'them': ['ui', 'ai', 'mai'],
    'you all': ['üi'],
}
R_VERB_TENSES = {
    'present': 'dü',
    'past': 'ku',
    'future': 'wei',
    'past_continuous': 'ti',
    'present_continuous': 'ti'
}
R_SUBJECT_PRONOUNS = {
    'i': ['nüü'],
    'you': ['üü'],
    'he': ['uhu', 'mahu'],
    'she': ['uhu', 'mahu'],
    'it': ['uhu', 'mahu'],
    'we': ['nüügwa', 'taagwa'],
    'they': ['uhuw̃a', 'mahuw̃a'],
    'you all': ['üügwa'],
    'this': ['ihi']
}
R_VERB_NOMINALIZERS = {
    'present': 'dü',
    'past': 'pü',
    'future': 'weidü',
}

def translate_simple(sentence: Dict[str, str]) -> Tuple[Subject, Verb, Object]:
    """Translate a simple English sentence to Paiute.

    Args:
        sentence (Dict[str, str]): A simple English sentence.

    Returns:
        List[Union[Subject, Verb, Object]]: A list of Paiute words.
    """
    sentence = {k: v.strip().lower() for k, v in sentence.items() if v}
    if sentence.get('object'):
        verb_stem = R_TRANSITIVE_VERBS.get(sentence['verb'], f"[{sentence['verb']}]")
    else:
        verb_stem = R_INTRANSITIVE_VERBS.get(sentence['verb'], R_TRANSITIVE_VERBS.get(sentence['verb'], f"[{sentence['verb']}]"))

    verb_tense = R_VERB_TENSES.get(sentence['verb_tense'], f"[{sentence['verb_tense']}]")
    # verb = f"{verb_stem}-{verb_tense}"
    verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=None)

    _object = None
    if (sentence.get('object') or '').strip(): # if there is an object
        if sentence['object'] in R_OBJECT_PRONOUNS:
            object_pronoun = random.choice(R_OBJECT_PRONOUNS.get(sentence['object'], [sentence['object']]))
            # verb = f"{object_pronoun}-{verb}"
            verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=object_pronoun)
        elif sentence.get('object_nominalizer'):
            _object = {**R_TRANSITIVE_VERBS, **R_INTRANSITIVE_VERBS}.get(sentence['object'], f"[{sentence['object']}]")
            object_suffix = random.choice(['ii', 'uu'])
            object_nominalizer = R_VERB_NOMINALIZERS.get(sentence['object_nominalizer'], f"[{sentence['object_nominalizer']}]")
            _object = Object(_object, object_nominalizer, object_suffix)
        else:
            _object = R_NOUNS.get(sentence['object'], f"[{sentence['object']}]")
            object_pronoun = random.choice(R_OBJECT_PRONOUNS['it'])
            object_suffix = Object.get_matching_suffix(object_pronoun)
            _object = Object(_object, None, object_suffix)
            # verb = f"{object_pronoun}-{verb}"
            verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=object_pronoun)

    if sentence.get('subject_nominalizer'):
        subject = {**R_TRANSITIVE_VERBS, **R_INTRANSITIVE_VERBS}.get(sentence['subject'], f"[{sentence['subject']}]")
        subject_suffix = random.choice(['ii', 'uu'])
        subject_nominalizer = R_VERB_NOMINALIZERS.get(sentence['subject_nominalizer'], f"[{sentence['subject_nominalizer']}]")
        # subject = f"{subject}-{subject_suffix}"
        subject = Subject(subject, subject_nominalizer, subject_suffix)
    elif sentence['subject'] in R_SUBJECT_PRONOUNS:
        subject = random.choice(R_SUBJECT_PRONOUNS.get(sentence['subject'], [sentence['subject']]))
        subject = Subject(subject, subject_noun_nominalizer=None, subject_suffix=None)
    else:
        subject = R_NOUNS.get(sentence['subject'], f"[{sentence['subject']}]")
        subject_suffix = random.choice(['ii', 'uu'])
        # subject = f"{subject}-{subject_suffix}"
        subject = Subject(subject, None, subject_suffix)

    return subject, verb, _object

def order_sentence(subject: Subject, verb: Verb, _object: Optional[Object] = None) -> List[Union[Subject, Verb, Object]]:
    if subject.noun in Subject.PRONOUNS:
        sentence = [subject, verb] if _object is None else [subject, _object, verb]
    else:
        sentence = [verb, subject] if _object is None else [_object, subject, verb]
    return sentence

def comparator_sentence(simple_sentence: Dict[str, str]) -> str:
    simple_sentence = {k: v.strip().lower() for k, v in simple_sentence.items() if v}
    subject_nominalizer = simple_sentence.get('subject_nominalizer')
    object_nominalizer = simple_sentence.get('object_nominalizer')
    if subject_nominalizer is None:
        if simple_sentence['subject'] not in {*R_SUBJECT_PRONOUNS, *R_NOUNS}:
            simple_sentence['subject'] = '[SUBJECT]'
    else:
        if simple_sentence['subject'] not in {*R_TRANSITIVE_VERBS, *R_INTRANSITIVE_VERBS}:
            simple_sentence['subject'] = '[SUBJECT]'
    if simple_sentence['verb'] not in R_TRANSITIVE_VERBS and simple_sentence['verb'] not in R_INTRANSITIVE_VERBS:
        simple_sentence['verb'] = '[VERB]'
    
    if object_nominalizer is None:
        if (simple_sentence.get('object') or '').strip() and simple_sentence['object'] not in {*R_NOUNS, *R_OBJECT_PRONOUNS}:
            simple_sentence['object'] = '[OBJECT]'
    else:
        if (simple_sentence.get('object') or '').strip() and simple_sentence['object'] not in R_NOUNS:
            simple_sentence['object'] = '[OBJECT]'

    return simple_sentence

def translate_english_to_ovp(sentence: str, model: str = None, res_callback: Optional[Callable[[ChatCompletion], None]] = None) -> Dict[str, Any]:
    simple_sentences = split_sentence(sentence, model=model, res_callback=res_callback)
    comparator_sentences = []
    target_simple_sentences = []
    backwards_translations = []
    for simple_sentence in simple_sentences:
        # print(simple_sentence)
        comparator_sentences.append(comparator_sentence(simple_sentence))
        # print(comparator_sentences[-1])
        subject, verb, _object = translate_simple(simple_sentence)
        # print(subject, verb, _object)
        target_simple_sentence = order_sentence(subject, verb, _object)
        target_simple_sentences.append(" ".join(map(str, target_simple_sentence)))
        backwards_translations.append(
            translate_ovp_to_english(
                subject_noun=subject.noun,
                subject_noun_nominalizer=subject.subject_noun_nominalizer,
                subject_suffix=subject.subject_suffix,
                verb=verb.verb_stem,
                verb_tense=verb.tense_suffix,
                object_pronoun=verb.object_pronoun_prefix,
                object_noun=_object.noun if _object else None,
                object_noun_nominalizer=_object.object_noun_nominalizer if _object else None,
                object_suffix=_object.object_suffix if _object else None
            ).strip(".")
        )
        # compare source sentence and com

    simple_sentences_nl = ". ".join([make_sentence(sentence, model=model, res_callback=res_callback) for sentence in simple_sentences]) + '.'
    comparator_sentence_nl = ". ".join([make_sentence(sentence, model=model, res_callback=res_callback) for sentence in comparator_sentences]) + '.'
    target_simple_sentence_nl = ". ".join(target_simple_sentences) + '.'
    backwards_translation_nl = ". ".join(backwards_translations) + '.'

    logging.info(f"Source: {sentence}")
    logging.info(f"Simple: {simple_sentences_nl}")
    logging.info(f"Comparator: {comparator_sentence_nl}")
    logging.info(f"Target: {target_simple_sentence_nl}")
    logging.info(f"BackTrans: {backwards_translation_nl}")

    sim_source_simple = semantic_similarity(sentence, simple_sentences_nl)
    logging.info(f"Source/Simple similarity: {sim_source_simple:0.3f}")
    # source/comparator similarity
    sim_source_comparator = semantic_similarity(sentence, comparator_sentence_nl)
    logging.info(f"Source/Comparator similarity: {sim_source_comparator:0.3f}")
    # source/backwards similarity
    sim_source_backwards = semantic_similarity(sentence, backwards_translation_nl)
    logging.info(f"Source/Backwards similarity: {sim_source_backwards:0.3f}")
    logging.info("--------")
    response = {
        "simple": simple_sentences_nl,
        "comparator": comparator_sentence_nl,
        "target": target_simple_sentence_nl,
        "backwards": backwards_translation_nl,
        "sim_simple": sim_source_simple,
        "sim_comparator": sim_source_comparator,
        "sim_backwards": sim_source_backwards
    }
    return response

def translate(sentence):
    logging.getLogger().setLevel(logging.ERROR)
    if sentence is None:
        while True:
            sentence = input("Enter a sentence: ")
            translation = translate_english_to_ovp(sentence)
            print(f"Target: {translation['target']}")
            print(f"Backwards: {translation['backwards']}")
            print()
    else:
        translation = translate_english_to_ovp(sentence)
        print(f"Target: {translation['target']}")
        print(f"Backwards: {translation['backwards']}")
        print()

def evaluate(models: List[str], max_tries: int = 15) -> None:
    path = thisdir / 'data' / 'sentences.csv'
    df = pd.read_csv(path)
    for model in models:
        path_similiarty = thisdir / '.results' / 'sentences-translated' / f"{model}.csv"
        path_similiarty.parent.mkdir(parents=True, exist_ok=True)
        if not path_similiarty.exists():
            df_similarity = df.copy()
            df_similarity['structure'] = None
            df_similarity['simple'] = None
            df_similarity['sim_simple'] = None
            df_similarity['comparator'] = None
            df_similarity['sim_comparator'] = None
            df_similarity['target'] = None
            df_similarity['backwards'] = None
            df_similarity['sim_backwards'] = None
        else:
            df_similarity = pd.read_csv(path_similiarty, index_col=0)

        os.environ['MODEL'] = model
        for i, sentence in enumerate(df_similarity['sentence']):
            if df_similarity.loc[df_similarity['sentence'] == sentence, 'sim_backwards'].notnull().values[0]:
                continue
            try_num = 1
            while True:
                try:
                    print(f"{i+1}/{len(df_similarity)}", end='\r')
                    simple_sentences = split_sentence(sentence)
                    comparator_sentences = []
                    target_simple_sentences = []
                    backwards_translations = []
                    
                    tokens = {'prompt': 0, 'completion': 0}
                    res_callback = lambda res: tokens.update({
                        'prompt': tokens['prompt'] + res.usage.prompt_tokens,
                        'completion': tokens['completion'] + res.usage.completion_tokens
                    })
                    for simple_sentence in simple_sentences:
                        comparator_sentences.append(comparator_sentence(simple_sentence))
                        subject, verb, _object = translate_simple(simple_sentence)
                        target_simple_sentence = order_sentence(subject, verb, _object)
                        target_simple_sentences.append(" ".join(map(str, target_simple_sentence)))
                        backwards_translation = translate_ovp_to_english(
                            subject_noun=subject.noun,
                            subject_suffix=subject.subject_suffix,
                            verb=verb.verb_stem,
                            verb_tense=verb.tense_suffix,
                            object_pronoun=verb.object_pronoun_prefix,
                            object_noun=_object.noun if _object else None,
                            object_suffix=_object.object_suffix if _object else None,
                            model=model,
                            res_callback=res_callback
                        )
                        backwards_translations.append(backwards_translation.strip("."))

                    simple_sentences_nl = ". ".join([make_sentence(sentence, model=model, res_callback=res_callback) for sentence in simple_sentences]) + '.'
                    comparator_sentence_nl = ". ".join([make_sentence(sentence, model=model, res_callback=res_callback) for sentence in comparator_sentences]) + '.'
                    target_simple_sentence_nl = ". ".join(target_simple_sentences) + '.'
                    backwards_translation_nl = ". ".join(backwards_translations) + '.'


                    df_similarity.loc[df_similarity['sentence'] == sentence, 'structure'] = json.dumps(simple_sentences)
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'simple'] = simple_sentences_nl
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'comparator'] = comparator_sentence_nl
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'target'] = target_simple_sentence_nl
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'backwards'] = backwards_translation_nl

                    df_similarity.loc[df_similarity['sentence'] == sentence, 'prompt_tokens'] = tokens['prompt']
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'completion_tokens'] = tokens['completion']

                    similarity_simple = semantic_similarity(sentence, simple_sentences_nl)
                    similarity_comparator = semantic_similarity(sentence, comparator_sentence_nl)
                    similarity_backwards = semantic_similarity(sentence, backwards_translation_nl)
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'sim_simple'] = similarity_simple
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'sim_comparator'] = similarity_comparator
                    df_similarity.loc[df_similarity['sentence'] == sentence, 'sim_backwards'] = similarity_backwards

                    df_similarity.to_csv(path_similiarty)
                    break
                except Exception as exc:
                    try_num += 1
                    if try_num > max_tries:
                        raise exc
                    traceback.print_exc()
                    print(f"Exception occurred ({exc}) - try {try_num}/{max_tries}")

def main():
    parser = argparse.ArgumentParser(description="Translate English to Paiute")
    subparsers = parser.add_subparsers(dest='command', required=True)

    translate_parser = subparsers.add_parser('translate', help="Translates sentences from English to Paiute")
    translate_parser.add_argument('sentence', help="The English sentence to translate (if not provided, will enter interactive mode)", nargs='?')
    translate_parser.set_defaults(func="translate")

    evaluate_parser = subparsers.add_parser('evaluate', help="Evaluate the translation of English sentences to Paiute")
    evaluate_parser.add_argument('models', nargs='+', help="Models to evaluate")
    evaluate_parser.set_defaults(func="evaluate")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    elif args.command == 'translate':
        translate(args.sentence)
    elif args.command == 'evaluate':
        evaluate(args.models)

if __name__ == '__main__':
    main()
