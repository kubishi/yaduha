from pydantic import BaseModel, Field, field_validator
from typing import Dict, Generator, List, Optional, Tuple, Type, Union
from enum import Enum
from random import choice, randint

from yaduha.language import Sentence, VocabEntry
from yaduha.language.zephyrian.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

# Lookup dictionaries for easy access
NOUN_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in NOUNS}
TRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in TRANSITIVE_VERBS}
INTRANSITIVE_VERB_LOOKUP: Dict[str, VocabEntry] = {entry.english: entry for entry in INTRANSITIVE_VERBS}


def get_noun_target(lemma: str) -> str:
    return NOUN_LOOKUP[lemma].target

def get_transitive_verb_target(lemma: str) -> str:
    return TRANSITIVE_VERB_LOOKUP[lemma].target

def get_intransitive_verb_target(lemma: str) -> str:
    return INTRANSITIVE_VERB_LOOKUP[lemma].target

def get_verb_target(lemma: str) -> str:
    if lemma in TRANSITIVE_VERB_LOOKUP:
        return TRANSITIVE_VERB_LOOKUP[lemma].target
    return INTRANSITIVE_VERB_LOOKUP[lemma].target


# ============================================================================
# VOWEL HARMONY SYSTEM
# Zephyrian uses vowel harmony - suffixes change based on the verb's vowel class
# Front vowels: e, i, y -> use front harmony suffixes
# Back vowels: a, o, u -> use back harmony suffixes
# ============================================================================

FRONT_VOWELS = {'e', 'i', 'y', 'ē', 'ī'}
BACK_VOWELS = {'a', 'o', 'u', 'ā', 'ō', 'ū'}

def get_dominant_vowel_class(word: str) -> str:
    """Determine if a word uses front or back vowel harmony"""
    for char in word:
        if char in FRONT_VOWELS:
            return "front"
        if char in BACK_VOWELS:
            return "back"
    return "back"  # default to back


def apply_harmony(suffix_front: str, suffix_back: str, base_word: str) -> str:
    """Apply vowel harmony to select the correct suffix variant"""
    if get_dominant_vowel_class(base_word) == "front":
        return suffix_front
    return suffix_back


# ============================================================================
# GRAMMATICAL ENUMERATIONS
# ============================================================================

class Evidentiality(str, Enum):
    """How the speaker knows the information"""
    witnessed = "witnessed"      # Speaker saw it directly
    reported = "reported"        # Speaker heard it from someone
    inferred = "inferred"        # Speaker deduced it from evidence

    def get_prefix(self) -> str:
        """Returns the evidential prefix for the verb"""
        if self == Evidentiality.witnessed:
            return "va"   # direct witness
        elif self == Evidentiality.reported:
            return "shi"  # hearsay
        else:
            return "zo"   # inference


class Tense(str, Enum):
    past = "past"
    present = "present"
    future = "future"

    def get_tone_marker(self, vowel_class: str) -> str:
        """Returns tonal suffix based on tense and vowel harmony"""
        if self == Tense.past:
            return "àn" if vowel_class == "back" else "èn"
        elif self == Tense.present:
            return "ā" if vowel_class == "back" else "ē"
        else:  # future
            return "úr" if vowel_class == "back" else "ír"


class Aspect(str, Enum):
    simple = "simple"           # Basic action
    continuous = "continuous"   # Ongoing action
    perfective = "perfective"   # Completed action

    def get_infix(self) -> str:
        """Returns aspect marker inserted after first consonant"""
        if self == Aspect.simple:
            return ""
        elif self == Aspect.continuous:
            return "el"
        else:
            return "or"


class Person(str, Enum):
    first = "first"
    second = "second"
    third = "third"


class Number(str, Enum):
    singular = "singular"
    plural = "plural"


class Animacy(str, Enum):
    """Zephyrian distinguishes animate and inanimate nouns"""
    animate = "animate"
    inanimate = "inanimate"

    def get_article(self, definite: bool) -> str:
        """Get the article based on animacy and definiteness"""
        if definite:
            return "el" if self == Animacy.animate else "ul"
        else:
            return "en" if self == Animacy.animate else "un"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Pronoun(BaseModel):
    person: Person
    number: Number

    def get_subject_form(self) -> str:
        """Get the subject pronoun form"""
        forms = {
            (Person.first, Number.singular): "zē",
            (Person.first, Number.plural): "zēn",
            (Person.second, Number.singular): "thū",
            (Person.second, Number.plural): "thūn",
            (Person.third, Number.singular): "vā",
            (Person.third, Number.plural): "vān",
        }
        return forms[(self.person, self.number)]

    def get_object_form(self) -> str:
        """Get the object pronoun form (uses different case)"""
        forms = {
            (Person.first, Number.singular): "mē",
            (Person.first, Number.plural): "mēn",
            (Person.second, Number.singular): "thē",
            (Person.second, Number.plural): "thēn",
            (Person.third, Number.singular): "lā",
            (Person.third, Number.plural): "lān",
        }
        return forms[(self.person, self.number)]


class Verb(BaseModel):
    lemma: str = Field(
        ...,
        json_schema_extra={
            'enum': [entry.english for entry in TRANSITIVE_VERBS + INTRANSITIVE_VERBS],
            'description': 'A verb lemma (transitive or intransitive)'
        }
    )
    tense: Tense
    aspect: Aspect = Aspect.simple
    evidentiality: Evidentiality = Evidentiality.witnessed

    @field_validator('lemma')
    @classmethod
    def validate_lemma(cls, v: str) -> str:
        if v not in TRANSITIVE_VERB_LOOKUP and v not in INTRANSITIVE_VERB_LOOKUP:
            raise ValueError(f"Invalid verb: {v}")
        return v

    def conjugate(self) -> str:
        """
        Conjugate the verb with evidentiality prefix, aspect infix, and tense suffix
        Structure: [evidential][aspect-modified stem][tense] (agglutinative, no separators)
        """
        stem = get_verb_target(self.lemma)
        vowel_class = get_dominant_vowel_class(stem)

        # Apply aspect infix after first consonant
        if self.aspect != Aspect.simple and len(stem) > 1:
            modified_stem = stem[0] + self.aspect.get_infix() + stem[1:]
        else:
            modified_stem = stem

        # Build: evidential + stem + tense (concatenated, no hyphens)
        evidential = self.evidentiality.get_prefix()
        tense_marker = self.tense.get_tone_marker(vowel_class)

        return f"{evidential}{modified_stem}{tense_marker}"


class TransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            'enum': [entry.english for entry in TRANSITIVE_VERBS],
            'description': 'A transitive verb lemma'
        }
    )


class IntransitiveVerb(Verb):
    lemma: str = Field(
        ...,
        json_schema_extra={
            'enum': [entry.english for entry in INTRANSITIVE_VERBS],
            'description': 'An intransitive verb lemma'
        }
    )


class Noun(BaseModel):
    head: str = Field(
        ...,
        json_schema_extra={
            'enum': [entry.english for entry in NOUNS],
            'description': 'A noun lemma'
        }
    )
    number: Number = Number.singular
    animacy: Animacy = Animacy.animate
    definite: bool = True

    @field_validator('head')
    @classmethod
    def validate_head(cls, v: str) -> str:
        if v not in NOUN_LOOKUP:
            raise ValueError(f"Invalid noun: {v}")
        return v

    def render(self) -> str:
        """
        Render the noun with article and number marking
        Structure: [article][noun][plural] (agglutinative, no separators)
        """
        base = get_noun_target(self.head)
        article = self.animacy.get_article(self.definite)

        # Plural suffix uses vowel harmony
        if self.number == Number.plural:
            plural_suffix = apply_harmony("ith", "ath", base)
            return f"{article}{base}{plural_suffix}"
        return f"{article}{base}"


class SubjectNoun(Noun):
    """Noun used as subject - no case marking needed in Zephyrian (VSO uses position)"""
    pass


class ObjectNoun(Noun):
    """Noun used as object - takes accusative prefix"""

    def render(self) -> str:
        """Objects take the 'ko' accusative prefix before the article"""
        base_render = super().render()
        return f"ko{base_render}"


# ============================================================================
# SENTENCE TYPES
# ============================================================================

class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    """
    Intransitive sentence: Verb-Subject order
    Example: "vasōmā elzāfir" = "The wolf sleeps" (witnessed)
    """
    verb: IntransitiveVerb
    subject: Union[SubjectNoun, Pronoun]

    def __str__(self) -> str:
        verb_str = self.verb.conjugate()

        if isinstance(self.subject, Pronoun):
            subject_str = self.subject.get_subject_form()
        else:
            subject_str = self.subject.render()

        # VSO order: Verb Subject
        return f"{verb_str} {subject_str}"

    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbSentence', None, None]:
        """Generate n sample sentences"""
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = Pronoun(
                    person=choice(list(Person)),
                    number=choice(list(Number))
                )
            else:
                subject = SubjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                    animacy=choice(list(Animacy)),
                    definite=choice([True, False])
                )

            verb = IntransitiveVerb(
                lemma=choice(list(INTRANSITIVE_VERB_LOOKUP.keys())),
                tense=choice(list(Tense)),
                aspect=choice(list(Aspect)),
                evidentiality=choice(list(Evidentiality))
            )

            yield cls(verb=verb, subject=subject)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbSentence"]]:
        return [
            (
                "I sleep.",
                SubjectVerbSentence(
                    verb=IntransitiveVerb(
                        lemma="sleep",
                        tense=Tense.present,
                        aspect=Aspect.simple,
                        evidentiality=Evidentiality.witnessed
                    ),
                    subject=Pronoun(
                        person=Person.first,
                        number=Number.singular
                    )
                )
            ),
            (
                "The wolf runs.",
                SubjectVerbSentence(
                    verb=IntransitiveVerb(
                        lemma="run",
                        tense=Tense.present,
                        aspect=Aspect.simple,
                        evidentiality=Evidentiality.witnessed
                    ),
                    subject=SubjectNoun(
                        head="wolf",
                        number=Number.singular,
                        animacy=Animacy.animate,
                        definite=True
                    )
                )
            ),
            (
                "The birds were flying (I heard).",
                SubjectVerbSentence(
                    verb=IntransitiveVerb(
                        lemma="fly",
                        tense=Tense.past,
                        aspect=Aspect.continuous,
                        evidentiality=Evidentiality.reported
                    ),
                    subject=SubjectNoun(
                        head="bird",
                        number=Number.plural,
                        animacy=Animacy.animate,
                        definite=True
                    )
                )
            ),
        ]


class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    """
    Transitive sentence: Verb-Subject-Object order
    Example: "vavēlē elzāfir koelthyra" = "The wolf sees the bird" (witnessed)
    """
    verb: TransitiveVerb
    subject: Union[SubjectNoun, Pronoun]
    object: Union[ObjectNoun, Pronoun]

    def __str__(self) -> str:
        verb_str = self.verb.conjugate()

        if isinstance(self.subject, Pronoun):
            subject_str = self.subject.get_subject_form()
        else:
            subject_str = self.subject.render()

        if isinstance(self.object, Pronoun):
            object_str = self.object.get_object_form()
        else:
            object_str = self.object.render()

        # VSO order: Verb Subject Object
        return f"{verb_str} {subject_str} {object_str}"

    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbObjectSentence', None, None]:
        """Generate n sample sentences"""
        for _ in range(n):
            if randint(0, 1) == 0:
                subject = Pronoun(
                    person=choice(list(Person)),
                    number=choice(list(Number))
                )
            else:
                subject = SubjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                    animacy=choice(list(Animacy)),
                    definite=choice([True, False])
                )

            verb = TransitiveVerb(
                lemma=choice(list(TRANSITIVE_VERB_LOOKUP.keys())),
                tense=choice(list(Tense)),
                aspect=choice(list(Aspect)),
                evidentiality=choice(list(Evidentiality))
            )

            if randint(0, 1) == 0:
                obj = Pronoun(
                    person=choice(list(Person)),
                    number=choice(list(Number))
                )
            else:
                obj = ObjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    number=choice(list(Number)),
                    animacy=choice(list(Animacy)),
                    definite=choice([True, False])
                )

            yield cls(verb=verb, subject=subject, object=obj)

    @classmethod
    def sample(cls, n: int) -> List['SubjectVerbObjectSentence']:
        return list(cls.sample_iter(n))

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        return [
            (
                "I see you.",
                SubjectVerbObjectSentence(
                    verb=TransitiveVerb(
                        lemma="see",
                        tense=Tense.present,
                        aspect=Aspect.simple,
                        evidentiality=Evidentiality.witnessed
                    ),
                    subject=Pronoun(
                        person=Person.first,
                        number=Number.singular
                    ),
                    object=Pronoun(
                        person=Person.second,
                        number=Number.singular
                    )
                )
            ),
            (
                "The cat ate the fish.",
                SubjectVerbObjectSentence(
                    verb=TransitiveVerb(
                        lemma="eat",
                        tense=Tense.past,
                        aspect=Aspect.perfective,
                        evidentiality=Evidentiality.witnessed
                    ),
                    subject=SubjectNoun(
                        head="cat",
                        number=Number.singular,
                        animacy=Animacy.animate,
                        definite=True
                    ),
                    object=ObjectNoun(
                        head="fish",
                        number=Number.singular,
                        animacy=Animacy.animate,
                        definite=True
                    )
                )
            ),
            (
                "They will read the books (I infer).",
                SubjectVerbObjectSentence(
                    verb=TransitiveVerb(
                        lemma="read",
                        tense=Tense.future,
                        aspect=Aspect.simple,
                        evidentiality=Evidentiality.inferred
                    ),
                    subject=Pronoun(
                        person=Person.third,
                        number=Number.plural
                    ),
                    object=ObjectNoun(
                        head="book",
                        number=Number.plural,
                        animacy=Animacy.inanimate,
                        definite=True
                    )
                )
            ),
        ]
