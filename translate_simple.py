"""Functions for translating simple sentences from English to Paiute."""
import json
import logging
import os
import pathlib
import random
import traceback
from typing import Dict, List, Optional, Tuple, Union

import dotenv
import numpy as np
import openai
import pandas as pd

from main import NOUNS, Object, Subject, Verb
from segment import make_sentence, split_sentence
from segment import semantic_similarity_sentence_transformers as semantic_similarity
from translate import translate as translate_paiute_to_english

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

thisdir = pathlib.Path(__file__).parent.absolute()

R_TRANSIITIVE_VERBS = {v: k for k, v in Verb.TRANSIITIVE_VERBS.items()}
R_INTRANSITIVE_VERBS = {v: k for k, v in Verb.INTRANSITIVE_VERBS.items()}
#     **{v: k for k, v in Verb.TRANSIITIVE_VERBS.items()},
#     **{v: k for k, v in Verb.INTRANSITIVE_VERBS.items()}
# }
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

def translate_simple(sentence: Dict[str, str]) -> Tuple[Subject, Verb, Object]:
    """Translate a simple English sentence to Paiute.

    Args:
        sentence (Dict[str, str]): A simple English sentence.

    Returns:
        List[Union[Subject, Verb, Object]]: A list of Paiute words.
    """
    sentence = {k: v.strip().lower() for k, v in sentence.items() if v}
    if sentence.get('object'):
        verb_stem = R_TRANSIITIVE_VERBS.get(sentence['verb'], f"[{sentence['verb']}]")
    else:
        verb_stem = R_INTRANSITIVE_VERBS.get(sentence['verb'], R_INTRANSITIVE_VERBS.get(sentence['verb'], f"[{sentence['verb']}]"))

    verb_tense = R_VERB_TENSES.get(sentence['verb_tense'], f"[{sentence['verb_tense']}]")
    # verb = f"{verb_stem}-{verb_tense}"
    verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=None)

    _object = None
    if (sentence.get('object') or '').strip(): # if there is an object
        if sentence['object'] in R_OBJECT_PRONOUNS:
            object_pronoun = random.choice(R_OBJECT_PRONOUNS.get(sentence['object'], [sentence['object']]))
            # verb = f"{object_pronoun}-{verb}"
            verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=object_pronoun)
        else:
            _object = R_NOUNS.get(sentence['object'], f"[{sentence['object']}]")
            object_pronoun = random.choice(R_OBJECT_PRONOUNS['it'])
            object_suffix = Object.get_matching_suffix(object_pronoun)
            _object = Object(_object, object_suffix)
            # verb = f"{object_pronoun}-{verb}"
            verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=object_pronoun)

    if sentence['subject'] in R_SUBJECT_PRONOUNS:
        subject = random.choice(R_SUBJECT_PRONOUNS.get(sentence['subject'], [sentence['subject']]))
        subject = Subject(subject, subject_suffix=None)
    else:
        subject = R_NOUNS.get(sentence['subject'], f"[{sentence['subject']}]")
        subject_suffix = random.choice(['ii', 'uu'])
        # subject = f"{subject}-{subject_suffix}"
        subject = Subject(subject, subject_suffix)

    return subject, verb, _object

def order_sentence(subject: Subject, verb: Verb, _object: Optional[Object] = None) -> List[Union[Subject, Verb, Object]]:
    if subject.noun in Subject.PRONOUNS:
        sentence = [subject, verb] if _object is None else [subject, _object, verb]
    else:
        sentence = [verb, subject] if _object is None else [_object, subject, verb]
    return sentence

def comparator_sentence(simple_sentence: Dict[str, str]) -> str:
    simple_sentence = {k: v.strip().lower() for k, v in simple_sentence.items() if v}
    if simple_sentence['subject'] not in R_NOUNS and simple_sentence['subject'] not in R_SUBJECT_PRONOUNS:
        simple_sentence['subject'] = '[SUBJECT]'
    if simple_sentence['verb'] not in R_TRANSIITIVE_VERBS and simple_sentence['verb'] not in R_INTRANSITIVE_VERBS:
        simple_sentence['verb'] = '[VERB]'
    if (simple_sentence.get('object') or '').strip() and simple_sentence['object'] not in R_NOUNS and simple_sentence['object'] not in R_OBJECT_PRONOUNS:
        simple_sentence['object'] = '[OBJECT]'
    return simple_sentence
    
def main():
    # set log level to error
    logging.getLogger().setLevel(logging.ERROR)
    while True:
        sentence = input("Enter a sentence: ")
        simple_sentences = split_sentence(sentence)
        comparator_sentences = []
        target_simple_sentences = []
        backwards_translations = []
        for simple_sentence in simple_sentences:
            comparator_sentences.append(comparator_sentence(simple_sentence))
            subject, verb, _object = translate_simple(simple_sentence)
            target_simple_sentence = order_sentence(subject, verb, _object)
            target_simple_sentences.append(" ".join(map(str, target_simple_sentence)))
            backwards_translations.append(
                translate_paiute_to_english(
                    subject_noun=subject.noun,
                    subject_suffix=subject.subject_suffix,
                    verb=verb.verb_stem,
                    verb_tense=verb.tense_suffix,
                    object_pronoun=verb.object_pronoun_prefix,
                    object_noun=_object.noun if _object else None,
                    object_suffix=_object.object_suffix if _object else None
                ).strip(".")
            )
            # compare source sentence and com

        simple_sentences_nl = ". ".join([make_sentence(sentence) for sentence in simple_sentences]) + '.'
        comparator_sentence_nl = ". ".join([make_sentence(sentence) for sentence in comparator_sentences]) + '.'
        target_simple_sentence_nl = ". ".join(target_simple_sentences) + '.'
        backwards_translation_nl = ". ".join(backwards_translations) + '.'

        print(f"Source: {sentence}")
        print(f"Simple: {simple_sentences_nl}")
        print(f"Comparator: {comparator_sentence_nl}")
        print(f"Target: {target_simple_sentence_nl}")
        print(f"BackTrans: {backwards_translation_nl}")

        
        sim_source_simple = semantic_similarity(sentence, simple_sentences_nl)
        print(f"Source/Simple similarity: {sim_source_simple:0.3f}")
        # source/comparator similarity
        sim_source_comparator = semantic_similarity(sentence, comparator_sentence_nl)
        print(f"Source/Comparator similarity: {sim_source_comparator:0.3f}")
        # source/backwards similarity
        sim_source_backwards = semantic_similarity(sentence, backwards_translation_nl)
        print(f"Source/Backwards similarity: {sim_source_backwards:0.3f}")
        print("--------")

def evaluate():
    path = thisdir / '.data' / 'sentences.csv'
    df = pd.read_csv(path)
    print(df)
    # count each "type"
    print(df['type'].value_counts())

    # get 100 random pairs of lines
    baseline_similarities = []
    for line1, line2 in zip(df.sample(100)['sentence'].values, df.sample(100)['sentence'].values):
        baseline_similarities.append(semantic_similarity(line1, line2))
        
    print(f"Mean baseline similarity: {np.mean(baseline_similarities):0.3f}")
    print(f"Variance: {np.var(baseline_similarities):0.3f}")

    path_similiarty = thisdir / '.data' / 'sentences-translated.csv'
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
    
    
    max_tries = 15
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
                for simple_sentence in simple_sentences:
                    comparator_sentences.append(comparator_sentence(simple_sentence))
                    subject, verb, _object = translate_simple(simple_sentence)
                    target_simple_sentence = order_sentence(subject, verb, _object)
                    target_simple_sentences.append(" ".join(map(str, target_simple_sentence)))
                    backwards_translations.append(
                        translate_paiute_to_english(
                            subject_noun=subject.noun,
                            subject_suffix=subject.subject_suffix,
                            verb=verb.verb_stem,
                            verb_tense=verb.tense_suffix,
                            object_pronoun=verb.object_pronoun_prefix,
                            object_noun=_object.noun if _object else None,
                            object_suffix=_object.object_suffix if _object else None
                        ).strip(".")
                    )
                    # compare source sentence and com

                simple_sentences_nl = ". ".join([make_sentence(sentence) for sentence in simple_sentences]) + '.'
                comparator_sentence_nl = ". ".join([make_sentence(sentence) for sentence in comparator_sentences]) + '.'
                target_simple_sentence_nl = ". ".join(target_simple_sentences) + '.'
                backwards_translation_nl = ". ".join(backwards_translations) + '.'

                df_similarity.loc[df_similarity['sentence'] == sentence, 'structure'] = json.dumps(simple_sentences)
                df_similarity.loc[df_similarity['sentence'] == sentence, 'simple'] = simple_sentences_nl
                df_similarity.loc[df_similarity['sentence'] == sentence, 'comparator'] = comparator_sentence_nl
                df_similarity.loc[df_similarity['sentence'] == sentence, 'target'] = target_simple_sentence_nl
                df_similarity.loc[df_similarity['sentence'] == sentence, 'backwards'] = backwards_translation_nl

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


if __name__ == '__main__':
    # main()
    evaluate()
