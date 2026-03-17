from pydantic import BaseModel, Field, field_validator
from typing import Dict, Generator, List, Optional, Tuple, Type, Union
from enum import Enum
from random import choice, randint

from yaduha.language import Language, Sentence, VocabEntry
from yaduha_ovp.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

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

LENIS_MAP = {
    'p': 'b',
    't': 'd',
    'k': 'g',
    's': 'z',
    'm': 'w̃'
}

def to_lenis(word: str) -> str:
    """Convert a word to its lenis form"""
    first_letter = word[0]
    if first_letter in LENIS_MAP:
        return LENIS_MAP[first_letter] + word[1:]
    else:
        return word


# ============================================================================
# GRAMMATICAL ENUMERATIONS
# ============================================================================

class Proximity(str, Enum):
    proximal = "proximal"
    distal = "distal"

    def get_object_suffix(self, does_end_in_glottal: bool) -> str:
        if self == Proximity.proximal:
            return "eika" if does_end_in_glottal else "neika"
        else:
            return "uka" if does_end_in_glottal else "noka"

    def get_subject_suffix(self) -> str:
        if self == Proximity.proximal:
            return "ii"
        else:
            return "uu"

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

class TenseAspect(str, Enum):
    past_simple = "past_simple"
    past_continuous = "past_continuous"
    present_perfect = "present_perfect"
    present_simple = "present_simple"
    present_continuous = "present_continuous"
    future_simple = "future_simple"

    def get_suffix(self) -> str:
        if self == TenseAspect.past_simple:
            return "ku"
        elif self in (TenseAspect.past_continuous, TenseAspect.present_continuous):
            return "ti"
        elif self == TenseAspect.present_perfect:
            return "pü"
        elif self == TenseAspect.present_simple:
            return "dü"
        elif self == TenseAspect.future_simple:
            return "wei"

        raise ValueError("Invalid tense/aspect combination")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Pronoun(BaseModel):
    person: Person
    plurality: Plurality
    proximity: Proximity
    inclusivity: Inclusivity
    reflexive: bool

    def get_subject_pronoun(self) -> str:
        if self.person == Person.first:
            if self.plurality == Plurality.singular:
                return 'nüü'
            elif self.plurality == Plurality.dual:
                return 'taa'
            elif self.plurality == Plurality.plural:
                if self.inclusivity == Inclusivity.inclusive:
                    return 'taagwa'
                else:
                    return 'nüügwa'
        elif self.person == Person.second:
            if self.plurality == Plurality.singular:
                return 'üü'
            else:
                return 'üügwa'
        elif self.person == Person.third:
            if self.plurality == Plurality.singular:
                if self.proximity == Proximity.proximal:
                    return 'mahu'
                else:
                    return 'uhu'
            else:
                if self.proximity == Proximity.proximal:
                    return 'mahuw̃a'
                else:
                    return 'uhuw̃a'

        raise ValueError("Invalid pronoun configuration")

    def get_object_pronoun(self) -> str:
        if self.person == Person.first:
            if self.plurality == Plurality.singular:
                return 'i'
            elif self.plurality == Plurality.dual:
                return 'ta'
            elif self.plurality == Plurality.plural:
                if self.inclusivity == Inclusivity.inclusive:
                    return 'tei'
                else:
                    return 'ni'
        elif self.person == Person.second:
            if self.plurality == Plurality.singular:
                return 'ü'
            else:
                return 'üi'
        elif self.person == Person.third:
            if self.reflexive:
                return 'na'
            if self.plurality == Plurality.singular:
                if self.proximity == Proximity.proximal:
                    return 'a'
                else:
                    return 'u'
            else:
                if self.proximity == Proximity.proximal:
                    return 'ai'
                else:
                    return 'ui'

        raise ValueError("Invalid pronoun configuration")

class Verb(BaseModel):
    lemma: str = Field(
        ...,
        json_schema_extra={
            'enum': [entry.english for entry in TRANSITIVE_VERBS + INTRANSITIVE_VERBS],
            'description': 'A verb lemma (transitive or intransitive)'
        }
    )
    tense_aspect: TenseAspect
    
    @field_validator('lemma')
    @classmethod
    def validate_lemma(cls, v: str) -> str:
        if v not in TRANSITIVE_VERB_LOOKUP and v not in INTRANSITIVE_VERB_LOOKUP:
            raise ValueError(f"Invalid verb: {v}")
        return v

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
    possessive_determiner: Optional[Pronoun] = None
    proximity: Proximity
    plurality: Plurality
    
    @field_validator('head')
    @classmethod
    def validate_head(cls, v: str) -> str:
        if v not in NOUN_LOOKUP:
            raise ValueError(f"Invalid noun: {v}")
        return v

class SubjectNoun(Noun):
    pass

class ObjectNoun(Noun):
    def get_matching_pronoun_prefix(self) -> str:
        return Pronoun(
            person=Person.third,
            plurality=self.plurality,
            proximity=self.proximity,
            inclusivity=Inclusivity.exclusive,
            reflexive=False
        ).get_object_pronoun()

class SubjectVerbSentence(Sentence["SubjectVerbSentence"]):
    subject: Union[SubjectNoun, Pronoun]
    verb: TransitiveVerb | IntransitiveVerb

    def __str__(self) -> str:
        subject_str = None
        if isinstance(self.subject, Pronoun):
            subject_str = self.subject.get_subject_pronoun()
        elif isinstance(self.subject, SubjectNoun):
            if isinstance(self.subject.head, Pronoun):
                subject_str = None
            else:
                target_word = get_noun_target(self.subject.head)
                subject_suffix = self.subject.proximity.get_subject_suffix()
                subject_str = f"{target_word}-{subject_suffix}"

        verb_stem = get_verb_target(self.verb.lemma)
        verb_suffix = self.verb.tense_aspect.get_suffix()
        verb_str = f"{verb_stem}-{verb_suffix}"

        return f"{subject_str} {verb_str}"
    
    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbSentence', None, None]:
        """Generate n sample sentences (string representations)"""
        for _ in range(n):
            # Random subject
            if randint(0, 1) == 0:
                subject = Pronoun(
                    person=choice(list(Person)),
                    plurality=choice(list(Plurality)),
                    proximity=choice(list(Proximity)),
                    inclusivity=choice(list(Inclusivity)),
                    reflexive=False
                )
            else:
                subject = SubjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    proximity=choice(list(Proximity)),
                    plurality=choice(list(Plurality))
                )

            # Random verb
            if randint(0, 1) == 0:
                verb = IntransitiveVerb(
                    lemma=choice(list(INTRANSITIVE_VERB_LOOKUP.keys())),
                    tense_aspect=choice(list(TenseAspect))
                )
            else:
                verb = TransitiveVerb(
                    lemma=choice(list(TRANSITIVE_VERB_LOOKUP.keys())),
                    tense_aspect=choice(list(TenseAspect))
                )

            yield cls(subject=subject, verb=verb)

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbSentence"]]:
        examples = [
            (
                "I sleep.",
                SubjectVerbSentence(
                    subject=Pronoun(
                        person=Person.first,
                        plurality=Plurality.singular,
                        proximity=Proximity.proximal,
                        inclusivity=Inclusivity.exclusive,
                        reflexive=False
                    ),
                    verb=IntransitiveVerb(
                        lemma="sleep",
                        tense_aspect=TenseAspect.present_simple
                    )
                )
            ),
            (
                "The coyote runs.",
                SubjectVerbSentence(
                    subject=SubjectNoun(
                        head="coyote",
                        proximity=Proximity.distal,
                        plurality=Plurality.singular
                    ),
                    verb=IntransitiveVerb(
                        lemma="run",
                        tense_aspect=TenseAspect.present_simple
                    )
                )
            ),
            (
                "The mountains will hit.",
                SubjectVerbSentence(
                    subject=SubjectNoun(
                        head="mountain",
                        proximity=Proximity.distal,
                        plurality=Plurality.plural
                    ),
                    verb=IntransitiveVerb(
                        lemma="hit",
                        tense_aspect=TenseAspect.future_simple
                    )
                )
            )
        ]

        return examples

class SubjectVerbObjectSentence(Sentence["SubjectVerbObjectSentence"]):
    subject: Union[SubjectNoun, Pronoun]
    verb: TransitiveVerb
    object: Union[ObjectNoun, Pronoun]

    def __str__(self) -> str:
        object_pronoun_prefix = None
        if isinstance(self.object, Pronoun):
            object_pronoun_prefix = self.object.get_object_pronoun()
        elif isinstance(self.object, ObjectNoun):
            object_pronoun_prefix = self.object.get_matching_pronoun_prefix()

        verb_stem = get_transitive_verb_target(self.verb.lemma) if self.object is not None else get_intransitive_verb_target(self.verb.lemma)
        verb_str = ""
        verb_suffix = self.verb.tense_aspect.get_suffix()
        verb_stem = to_lenis(verb_stem)
        verb_str = f"{object_pronoun_prefix}-{verb_stem}-{verb_suffix}"

        object_str = None
        if isinstance(self.object, ObjectNoun):
            target_word = get_noun_target(self.object.head)
            does_end_in_glottal = target_word.endswith("'")
            object_suffix = self.object.proximity.get_object_suffix(does_end_in_glottal)
            object_str = f"{target_word}-{object_suffix}"

        subject_str = None
        if isinstance(self.subject, Pronoun):
            subject_str = self.subject.get_subject_pronoun()
        elif isinstance(self.subject, SubjectNoun):
            if isinstance(self.subject.head, Pronoun):
                subject_str = None
            else:
                target_word = get_noun_target(self.subject.head)
                subject_suffix = self.subject.proximity.get_subject_suffix()
                subject_str = f"{target_word}-{subject_suffix}"

        if object_str is None:
            return f"{verb_str} {subject_str}"
        else:
            return f"{subject_str} {object_str} {verb_str}"

    @classmethod
    def sample_iter(cls, n: int) -> Generator['SubjectVerbObjectSentence', None, None]:
        """Generate n sample sentences (string representations)"""
        for _ in range(n):
            # Random subject
            if randint(0, 1) == 0:
                subject = Pronoun(
                    person=choice(list(Person)),
                    plurality=choice(list(Plurality)),
                    proximity=choice(list(Proximity)),
                    inclusivity=choice(list(Inclusivity)),
                    reflexive=False
                )
            else:
                subject = SubjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    proximity=choice(list(Proximity)),
                    plurality=choice(list(Plurality))
                )

            # Random verb
            verb_lemma = choice(list(TRANSITIVE_VERB_LOOKUP.keys()))
            verb = TransitiveVerb(
                lemma=verb_lemma,
                tense_aspect=choice(list(TenseAspect))
            )

            # Random object for transitive verbs
            if randint(0, 1) == 0:
                obj = ObjectNoun(
                    head=choice(list(NOUN_LOOKUP.keys())),
                    proximity=choice(list(Proximity)),
                    plurality=choice(list(Plurality))
                )
            else:
                obj = Pronoun(
                    person=choice(list(Person)),
                    plurality=choice(list(Plurality)),
                    proximity=choice(list(Proximity)),
                    inclusivity=choice(list(Inclusivity)),
                    reflexive=False
                )

            yield cls(subject=subject, verb=verb, object=obj)

    @classmethod
    def sample(cls, n: int) -> List['SubjectVerbObjectSentence']:
        """Generate n sample sentences (string representations)"""
        return list(cls.sample_iter(n))

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SubjectVerbObjectSentence"]]:
        examples = [
            (
                "You read the mountains.",
                SubjectVerbObjectSentence(
                    subject=Pronoun(
                        person=Person.second,
                        plurality=Plurality.singular,
                        proximity=Proximity.distal,
                        inclusivity=Inclusivity.exclusive,
                        reflexive=False
                    ),
                    verb=TransitiveVerb(
                        lemma="read",
                        tense_aspect=TenseAspect.present_simple
                    ),
                    object=ObjectNoun(
                        head="mountain",
                        proximity=Proximity.distal,
                        plurality=Plurality.plural
                    )
                ),
            ),
            (
                "That worm will hear it.",
                SubjectVerbObjectSentence(
                    subject=SubjectNoun(
                        head="worm",
                        proximity=Proximity.distal,
                        plurality=Plurality.singular
                    ),
                    verb=TransitiveVerb(
                        lemma="hear",
                        tense_aspect=TenseAspect.future_simple
                    ),
                    object=Pronoun(
                        person=Person.third,
                        plurality=Plurality.singular,
                        proximity=Proximity.distal,
                        inclusivity=Inclusivity.exclusive,
                        reflexive=False
                    )
                )
            ),
            (
                "That food cooks this weasle.",
                SubjectVerbObjectSentence(
                    subject=SubjectNoun(
                        head="food",
                        proximity=Proximity.distal,
                        plurality=Plurality.singular
                    ),
                    verb=TransitiveVerb(
                        lemma="cook",
                        tense_aspect=TenseAspect.present_simple
                    ),
                    object=ObjectNoun(
                        head="weasle",
                        proximity=Proximity.proximal,
                        plurality=Plurality.singular
                    )
                )
            )
        ]

        return examples


language = Language(
    code="ovp",
    name="Owens Valley Paiute",
    sentence_types=(SubjectVerbSentence, SubjectVerbObjectSentence),
)
