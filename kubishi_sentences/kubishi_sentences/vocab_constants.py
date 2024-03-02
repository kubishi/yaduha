import json
import logging
import pathlib
import random
from itertools import starmap
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

NOUNS = {
    "isha'": "coyote",
    "isha'pugu": "dog",
    "kidi'": "cat",
    "pugu": "horse",
    "wai": "rice",
    "tüba": "pinenuts",
    "maishibü": "corn",
    "paya": "water",
    "payahuupü": "river",
    "katünu": "chair",
    "toyabi": "mountain",
    "tuunapi": "food",
    "pasohobü": "tree",
    "nobi": "house",
    "toni": "wickiup",
    "apo": "cup",
    "küna": "wood",
    "tübbi": "rock",
    "tabuutsi'": "cottontail",
    "kamü": "jackrabbit",
    "aaponu'": "apple",
    "tüsüga": "weasle",
    "mukita": "lizard",
    "wo'ada": "mosquito",
    "wükada": "bird snake",
    "wo'abi": "worm",
    "aingwü": "squirrel",
    "tsiipa": "bird",
    "tüwoobü": "earth",
    "koopi'": "coffee",
    "pahabichi": "bear",
    "pagwi": "fish",
    "kwadzi": "tail",
}

LENIS_MAP = {"p": "b", "t": "d", "k": "g", "s": "z", "m": "w̃"}


class Subject:
    SUFFIXES = {
        "ii": "proximal",
        "uu": "distal",
    }
    PRONOUNS = {
        "nüü": "I",
        "uhu": "he/she/it",
        "uhuw̃a": "they",
        "mahu": "he/she/it",
        "mahuw̃a": "they",
        "ihi": "this",
        "ihiw̃a": "these",
        "taa": "you and I",
        "nüügwa": "we (exclusive)",
        "taagwa": "we (inclusive)",
        "üü": "you",
        "üügwa": "you (plural)",
    }

    def __init__(self, noun: str, subject_suffix: Optional[str]):
        self.noun = noun
        self.subject_suffix = subject_suffix

        if self.noun in Subject.PRONOUNS and self.subject_suffix is not None:
            raise ValueError("Subject suffix is not allowed with pronouns")

        if self.noun not in Subject.PRONOUNS:
            if self.subject_suffix is None:
                raise ValueError("Subject suffix is required with non-pronoun subjects")

            if subject_suffix not in self.SUFFIXES:
                raise ValueError(f"Subject suffix must be one of {self.SUFFIXES} (not {subject_suffix})")

    def __str__(self) -> str:
        if self.subject_suffix is None:
            return self.noun
        else:
            return f"{self.noun}-{self.subject_suffix}"

    @property
    def details(self) -> Dict:
        data = {"type": "subject", "text": str(self), "parts": []}
        if self.noun in Subject.PRONOUNS:
            data["parts"].append({"type": "pronoun", "text": self.noun, "definition": Subject.PRONOUNS[self.noun]})
        else:
            data["parts"].append({"type": "noun", "text": self.noun, "definition": NOUNS[self.noun]})
            data["parts"].append(
                {
                    "type": "subject_suffix",
                    "text": self.subject_suffix,
                    "definition": self.SUFFIXES[self.subject_suffix],
                }
            )
        return data


class Verb:
    TENSES = {
        "ku": "completive (past)",
        "ti": "present ongoing (-ing)",
        "dü": "present",
        "wei": "future (will)",
        "gaa-wei": "future (going to)",
        "pü": "have x-ed, am x-ed",
    }
    TRANSIITIVE_VERBS = {
        "tüka": "eat",
        "puni": "see",
        "hibi": "drink",
        "naka": "hear",
        "kwana": "smell",
        "kwati": "hit",
        "yadohi": "talk to",
        "naki": "chase",
        "tsibui": "climb",
        "sawa": "cook",
        "tama'i": "find",
        "nia": "read",
        "mui": "write",
        "nobini": "visit",
    }
    INTRANSITIVE_VERBS = {
        "katü": "sit",
        "üwi": "sleep",
        "kwisha'i": "sneeze",
        "poyoha": "run",
        "mia": "go",
        "hukaw̃ia": "walk",
        "wünü": "stand",
        "habi": "lie down",
        "yadoha": "talk",
        "kwatsa'i": "fall",
        "waakü": "work",
        "wükihaa": "smile",
        "hubiadu": "sing",
        "nishua'i": "laugh",
        "tsibui": "climb",
        "tübinohi": "play",
        "yotsi": "fly",
        "nüga": "dance",
        "pahabi": "swim",
        "tünia": "read",
        "tümui": "write",
        "tsiipe'i": "chirp",
    }

    def __init__(self, verb_stem: str, tense_suffix: str, object_pronoun_prefix: Optional[str]):
        self.verb_stem = verb_stem
        self.tense_suffix = tense_suffix
        self.object_pronoun_prefix = object_pronoun_prefix

        if tense_suffix not in self.TENSES:
            raise ValueError(f"Tense must be one of {self.TENSES} (not {tense_suffix})")

        if self.verb_stem in Verb.TRANSIITIVE_VERBS:
            # if self.object_pronoun_prefix is None:
            #     raise ValueError("Object pronoun prefix is required for transitive verbs")
            pass
        elif self.verb_stem in Verb.INTRANSITIVE_VERBS:
            if self.object_pronoun_prefix is not None:
                raise ValueError("Intransitive verbs cannot have object pronouns")
        else:
            # raise ValueError(f"Verb stem must be one of {Verb.TRANSIITIVE_VERBS} or {Verb.INTRANSITIVE_VERBS} (not {verb_stem})")
            logging.warning(
                f"Verb stem must be one of {Verb.TRANSIITIVE_VERBS} or {Verb.INTRANSITIVE_VERBS} (not {verb_stem})"
            )

    def __str__(self) -> str:
        if self.object_pronoun_prefix is None:
            return f"{self.verb_stem}-{self.tense_suffix}"
        else:
            verb_stem = to_lenis(self.verb_stem)
            return f"{self.object_pronoun_prefix}-{verb_stem}-{self.tense_suffix}"

    @property
    def is_transitive(self) -> bool:
        return self.verb_stem in Verb.TRANSIITIVE_VERBS

    @property
    def details(self) -> Dict:
        data = {"type": "verb", "text": str(self), "parts": []}
        if self.object_pronoun_prefix is not None:
            data["parts"].append(
                {
                    "type": "object_pronoun",
                    "text": self.object_pronoun_prefix,
                    "definition": Object.PRONOUNS[self.object_pronoun_prefix],
                }
            )
        data["parts"].append(
            {
                "type": "verb_stem",
                "text": self.verb_stem,
                "definition": (
                    Verb.TRANSIITIVE_VERBS[self.verb_stem]
                    if self.is_transitive
                    else Verb.INTRANSITIVE_VERBS[self.verb_stem]
                ),
            }
        )
        data["parts"].append({"type": "tense", "text": self.tense_suffix, "definition": self.TENSES[self.tense_suffix]})
        return data


class Object:
    SUFFIXES = {
        "eika": "proximal",
        "oka": "distal",
    }
    PRONOUNS = {
        "i": "me",
        "u": "him/her/it (distal)",
        "ui": "them (distal)",
        "ma": "him/her/it (proximal)",
        "mai": "them (proximal)",
        "a": "him/her/it (proximal)",
        "ai": "them (proximal)",
        "ni": "us (plural, exclusive)",
        "tei": "us (plural, inclusive)",
        "ta": "us (dual), you and I",
        "ü": "you (singular)",
        "üi": "you (plural), you all",
    }

    def __init__(self, noun: str, object_suffix: Optional[str]):
        self.noun = noun
        self.object_suffix = object_suffix

        if self.object_suffix is None:
            raise ValueError("Object suffix is required")
        elif self.object_suffix not in self.SUFFIXES:
            raise ValueError(f"Object suffix must be one of {self.SUFFIXES} (not {object_suffix})")

    def __str__(self) -> str:
        object_suffix = self.object_suffix
        if "'" not in self.noun[-2:]:  # noun does not end in glottal stop
            if object_suffix == "eika":
                object_suffix = "neika"
            elif object_suffix == "oka":
                object_suffix = "noka"
        return f"{self.noun}-{object_suffix}"

    @classmethod
    def check_agreement(self, object_suffix: str, object_pronoun: str) -> bool:
        """Check whether the object suffix and object pronoun agree

        Either both must contain 'proximal', both contain 'distal', or neither contain either.
        """
        if "proximal" in object_suffix and "proximal" in object_pronoun:
            return True
        elif "distal" in object_suffix and "distal" in object_pronoun:
            return True
        elif (
            "proximal" not in object_suffix
            and "proximal" not in object_pronoun
            and "distal" not in object_suffix
            and "distal" not in object_pronoun
        ):
            return True
        else:
            return False

    @classmethod
    def get_matching_suffix(self, object_pronoun: str) -> str:
        """Get the object suffix that matches the object pronoun

        If the object pronoun is proximal, return the proximal suffix.
        If the object pronoun is distal, return the distal suffix.
        """
        if object_pronoun is None:
            return None
        elif "proximal" in Object.PRONOUNS[object_pronoun]:
            return "eika"
        elif "distal" in Object.PRONOUNS[object_pronoun]:
            return "oka"
        else:
            return None

    @classmethod
    def get_matching_third_person_pronouns(self, object_suffix: str) -> List[str]:
        """Get the object pronouns that match the object suffix

        If the object suffix is proximal, return the proximal pronouns.
        If the object suffix is distal, return the distal pronouns.
        """
        proximal_pronouns = [pronoun for pronoun in self.PRONOUNS if "proximal" in self.PRONOUNS[pronoun]]
        distal_pronouns = [pronoun for pronoun in self.PRONOUNS if "distal" in self.PRONOUNS[pronoun]]
        third_person_pronouns = [*proximal_pronouns, *distal_pronouns]
        if not object_suffix:
            return third_person_pronouns
        elif object_suffix == "eika":
            return proximal_pronouns
        elif object_suffix == "oka":
            return distal_pronouns
        else:
            raise ValueError(f"Object suffix must be one of {self.SUFFIXES}")

    @property
    def details(self) -> Dict:
        data = {"type": "object", "text": str(self), "parts": []}
        data["parts"].append({"type": "noun", "text": self.noun, "definition": NOUNS[self.noun]})
        data["parts"].append(
            {"type": "object_suffix", "text": self.object_suffix, "definition": self.SUFFIXES[self.object_suffix]}
        )
        return data
