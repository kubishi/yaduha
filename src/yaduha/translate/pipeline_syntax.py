from pydantic import BaseModel, ValidationError
from typing import List, Optional, Union
from enum import Enum




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
