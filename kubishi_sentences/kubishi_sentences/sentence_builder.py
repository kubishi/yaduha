import json
import logging
import pathlib
import random
from itertools import starmap
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from kubishi_sentences.vocab_constants import NOUNS, Object, Subject, Verb


def to_lenis(word: str) -> str:
    """Convert a word to its lenis form"""
    first_letter = word[0]
    if first_letter in LENIS_MAP:
        return LENIS_MAP[first_letter] + word[1:]
    else:
        return word


def get_all_choices(
    subject_noun: Optional[str],
    subject_suffix: Optional[str],
    verb: Optional[str],
    verb_tense: Optional[str],
    object_pronoun: Optional[str],
    object_noun: Optional[str],
    object_suffix: Optional[str],
) -> Dict[str, Any]:
    choices = {}
    # Validate inputs
    if subject_noun not in {*Subject.PRONOUNS.keys(), *NOUNS.keys()}:
        subject_noun = None

    if subject_suffix not in Subject.SUFFIXES.keys():
        subject_suffix = None

    if verb not in [*Verb.TRANSIITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()]:
        verb = None

    if verb_tense not in Verb.TENSES.keys():
        verb_tense = None

    if object_pronoun not in Object.PRONOUNS.keys():
        object_pronoun = None

    if object_noun not in NOUNS.keys():
        object_noun = None

    if object_suffix not in Object.SUFFIXES.keys():
        object_suffix = None

    # Check object_pronoun and object_suffix match
    # if mismatch, set to None (will be corrected below)
    if object_pronoun is not None and object_suffix is not None:
        if object_pronoun not in Object.get_matching_third_person_pronouns(object_suffix):
            object_suffix = None

    def to_choice(word: str, trans: str) -> Tuple[str, str]:
        return word, f"{word}: {trans}"

    # Subject
    choices["subject_noun"] = {
        # 'choices': [*Subject.PRONOUNS.keys(), *NOUNS.keys()],
        "choices": [*starmap(to_choice, Subject.PRONOUNS.items()), *starmap(to_choice, NOUNS.items())],
        "value": subject_noun,
        "requirement": "required",
    }
    if subject_noun is None:
        choices["subject_suffix"] = {"choices": [], "value": None, "requirement": "disabled"}
    elif subject_noun in Subject.PRONOUNS:
        choices["subject_suffix"] = {"choices": [], "value": None, "requirement": "disabled"}
    else:
        choices["subject_suffix"] = {
            # 'choices': list(Subject.SUFFIXES.keys()),
            "choices": [*starmap(to_choice, Subject.SUFFIXES.items())],
            "value": subject_suffix,
            "requirement": "required",
        }

    # Verb
    if object_noun is not None:  # verb must be transitive
        choices["verb"] = {
            # 'choices': list(Verb.TRANSIITIVE_VERBS.keys()),
            "choices": [*starmap(to_choice, Verb.TRANSIITIVE_VERBS.items())],
            "value": verb,
            "requirement": "required",
        }
    else:
        choices["verb"] = {
            # 'choices': [*Verb.TRANSIITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()],
            "choices": [
                *starmap(to_choice, Verb.TRANSIITIVE_VERBS.items()),
                *starmap(to_choice, Verb.INTRANSITIVE_VERBS.items()),
            ],
            "value": verb,
            "requirement": "required",
        }

    # Verb tense
    if verb is None:
        choices["verb_tense"] = {"choices": [], "value": None, "requirement": "disabled"}
    else:
        choices["verb_tense"] = {
            # 'choices': list(Verb.TENSES.keys()),
            "choices": [*starmap(to_choice, Verb.TENSES.items())],
            "value": verb_tense,
            "requirement": "required",
        }

    # Object pronoun
    if verb is None or verb in Verb.INTRANSITIVE_VERBS:
        choices["object_pronoun"] = {"choices": [], "value": None, "requirement": "disabled"}
    elif object_noun is not None:  # object pronoun must match object suffix
        choices["object_pronoun"] = {
            "choices": [
                to_choice(pronoun, Object.PRONOUNS[pronoun])
                for pronoun in Object.get_matching_third_person_pronouns(object_suffix)
            ],
            "value": object_pronoun,
            "requirement": "required",
        }
    else:
        choices["object_pronoun"] = {
            # 'choices': list(Object.PRONOUNS.keys()),
            "choices": [*starmap(to_choice, Object.PRONOUNS.items())],
            "value": object_pronoun,
            "requirement": "optional",
        }

    # Object noun
    # if verb is intransitive, object noun must be None
    # if verb in Verb.INTRANSITIVE_VERBS or object_pronoun not in Object.get_matching_third_person_pronouns(None):
    if verb in Verb.INTRANSITIVE_VERBS or object_pronoun not in [
        None,
        *Object.get_matching_third_person_pronouns(None),
    ]:
        choices["object_noun"] = {"choices": [], "value": None, "requirement": "disabled"}
    else:  # verb is not selected or is transitive
        choices["object_noun"] = {
            # 'choices': list(NOUNS.keys()),
            "choices": [*starmap(to_choice, NOUNS.items())],
            "value": object_noun,
            "requirement": "required",
        }

    # Object suffix
    if object_noun is None:
        choices["object_suffix"] = {"choices": [], "value": None, "requirement": "disabled"}
    elif object_pronoun is not None:
        matching_suffix = Object.get_matching_suffix(object_pronoun)
        choices["object_suffix"] = {
            "choices": (
                [] if matching_suffix is None else [to_choice(matching_suffix, Object.SUFFIXES[matching_suffix])]
            ),
            "value": None if object_suffix != Object.get_matching_suffix(object_pronoun) else object_suffix,
            "requirement": "required",
        }
    else:
        choices["object_suffix"] = {
            # 'choices': list(Object.SUFFIXES.keys()),
            "choices": [*starmap(to_choice, Object.SUFFIXES.items())],
            "value": object_suffix,
            "requirement": "required",
        }

    return choices


def format_sentence(
    subject_noun: Optional[str],
    subject_suffix: Optional[str],
    verb: Optional[str],
    verb_tense: Optional[str],
    object_pronoun: Optional[str],
    object_noun: Optional[str],
    object_suffix: Optional[str],
) -> List[Dict]:
    subject = Subject(subject_noun, subject_suffix)
    _verb = Verb(verb, verb_tense, object_pronoun)

    # check object_pronoun and object_suffix match
    if object_suffix is not None:
        if object_pronoun not in Object.get_matching_third_person_pronouns(object_suffix):
            raise ValueError("Object pronoun and suffix do not match")

    object = None
    try:
        object = Object(object_noun, object_suffix)
    except ValueError as e:  # could not create object
        if object_noun is not None:
            raise e
        else:  # okay, since object is optional
            pass

    if subject.noun in Subject.PRONOUNS:
        if object:
            return [object.details, subject.details, _verb.details]
        else:
            return [_verb.details, subject.details]
    else:
        if object:
            return [subject.details, object.details, _verb.details]
        else:
            return [subject.details, _verb.details]


def get_random_sentence(choices: Dict[str, Dict[str, Any]] = {}):
    if not choices:
        choices = get_all_choices(None, None, None, None, None, None, None)
    all_keys = list(choices.keys())
    i = 0
    while True:
        random.shuffle(all_keys)
        for key in all_keys:
            if not choices[key]["choices"] or choices[key]["value"]:
                continue
            choices[key]["value"], _ = random.choice(choices[key]["choices"])
            choices = get_all_choices(**{k: v["value"] for k, v in choices.items()})
            i = 0

        try:
            format_sentence(**{k: v["value"] for k, v in choices.items()})
            return choices
        except ValueError as e:
            i += 1
            if i > 20:
                raise e
            continue


def get_random_sentence_big():
    subject_noun = random.choice(list(NOUNS.keys()))
    subject_suffix = random.choice(list(Subject.SUFFIXES.keys()))
    verb = random.choice(list(Verb.TRANSIITIVE_VERBS.keys()))
    verb_tense = random.choice(list(Verb.TENSES.keys()))
    object_pronoun = random.choice(list(Object.get_matching_third_person_pronouns(None)))
    object_noun = random.choice(list(NOUNS.keys()))
    object_suffix = Object.get_matching_suffix(object_pronoun)

    choices = get_all_choices(
        subject_noun, subject_suffix, verb, verb_tense, object_pronoun, object_noun, object_suffix
    )
    return choices


def sentence_to_str(sentence: List[Dict]):
    text = ""
    for word in sentence:
        text += word["text"] + " "
    return text


def print_sentence(sentence: List[Dict]):
    print(sentence_to_str(sentence))


def main():
    for _ in range(100):
        choices = get_random_sentence()
        sentence = format_sentence(**{k: v["value"] for k, v in choices.items()})
        print_sentence(sentence)


if __name__ == "__main__":
    main()
