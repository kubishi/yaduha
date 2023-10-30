"""Functions for translating simple sentences from English to Paiute."""
import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import dotenv
import openai

from segment import split_sentence, make_sentence, semantic_similarity_spacy
from main import NOUNS, Verb, Object, Subject
from translate import translate as translate_paiute_to_english

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')


R_VERBS = {
    **{v: k for k, v in Verb.TRANSIITIVE_VERBS.items()},
    **{v: k for k, v in Verb.INTRANSITIVE_VERBS.items()}
}
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
    'they': ['uhuwa', 'mahuwa'],
    'you all': ['üügwa'],
}

def translate_simple(sentence: Dict[str, str]) -> Tuple[Subject, Verb, Object]:
    """Translate a simple English sentence to Paiute.

    Args:
        sentence (Dict[str, str]): A simple English sentence.

    Returns:
        List[Union[Subject, Verb, Object]]: A list of Paiute words.
    """
    sentence = {k: v.strip().lower() for k, v in sentence.items()}
    verb_stem = R_VERBS.get(sentence['verb'], f"[{sentence['verb']}]")

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
    simple_sentence = {k: v.strip().lower() for k, v in simple_sentence.items()}
    if simple_sentence['subject'] not in R_NOUNS and simple_sentence['subject'] not in R_SUBJECT_PRONOUNS:
        simple_sentence['subject'] = '[SUBJECT]'
    if simple_sentence['verb'] not in R_VERBS:
        simple_sentence['verb'] = '[VERB]'
    if (simple_sentence.get('object') or '').strip() and simple_sentence['object'] not in R_NOUNS and simple_sentence['object'] not in R_OBJECT_PRONOUNS:
        simple_sentence['object'] = '[OBJECT]'
    return simple_sentence
    
def main():
    sentence = "The dog was running yesterday and fell while watching an armadillo."
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

    # source/simple similarity
    sim_source_simple = semantic_similarity_spacy(sentence, simple_sentences_nl)
    print(f"Source/Simple similarity: {sim_source_simple:0.3f}")
    # source/comparator similarity
    sim_source_comparator = semantic_similarity_spacy(sentence, comparator_sentence_nl)
    print(f"Source/Comparator similarity: {sim_source_comparator:0.3f}")
    # source/backwards similarity
    sim_source_backwards = semantic_similarity_spacy(sentence, backwards_translation_nl)
    print(f"Source/Backwards similarity: {sim_source_backwards:0.3f}")

if __name__ == '__main__':
    main()