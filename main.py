

import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


NOUNS = {
    "isha'": "coyote",
    "isha'pugu": "dog",
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
}

class Subject:
    SUFFIXES = {
        'ii': 'proximal',
        'uu': 'distal',
    }
    PRONOUNS = {
        "nüü": "I",
        "uhu": "he/she/it (distal)",
        "uhuw̃a": 'they (distal)',
        "mahu": "he/she/it (proximal)",
        "mahuw̃a": 'they (proximal)',
        "ihi": "this",
        "ihiw̃a": "these",
        "taa": "we (dual), you and I",
        "nüügwa": "we (plural, exclusive)",
        "taagwa": "we (plural, inclusive)",
        "üü": "you (singular)",
        "üügwa": "you (plural), you all",
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
        data = {
            'type': 'subject',
            'text': str(self),
            'parts': []
        }
        if self.noun in Subject.PRONOUNS:
            data['parts'].append({
                'type': 'pronoun',
                'text': self.noun,
                'definition': Subject.PRONOUNS[self.noun]
            })
        else:
            data['parts'].append({
                'type': 'noun',
                'text': self.noun,
                'definition': NOUNS[self.noun]
            })
            data['parts'].append({
                'type': 'subject_suffix',
                'text': self.subject_suffix,
                'definition': self.SUFFIXES[self.subject_suffix]
            })
        return data


LENIS_MAP = {
    'p': 'b',
    't': 'd',
    'k': 'g',
    's': 'z',
    'm': 'w̃'
}       
def to_lenis(word: str) -> str:
    """Convert a word to its lenis form
    """
    first_letter = word[0]
    if first_letter in LENIS_MAP:
        return LENIS_MAP[first_letter] + word[1:]
    else:
        return word

class Verb:
    TENSES = {
        'ku': 'completive',
        'ti': 'present ongoing',
        'dü': 'agent nominalizer, present',
        'wei': 'future (will)',
        'gaa-wei': 'future (going to)',
        'pü': 'present perfect'
    }
    TRANSIITIVE_VERBS = {
        'tüka': 'to eat',
        'puni': 'to see',
    }
    INTRANSITIVE_VERBS = {
        'katü': 'to sit',
        'üwi': 'to sleep',
    }
    def __init__(self, 
                 verb_stem: str, 
                 tense_suffix: str, 
                 object_pronoun_prefix: Optional[str]):
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
            raise ValueError(f"Verb stem must be one of {Verb.TRANSIITIVE_VERBS} or {Verb.INTRANSITIVE_VERBS} (not {verb_stem})")
        
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
        data = {
            'type': 'verb',
            'text': str(self),
            'parts': []
        }
        if self.object_pronoun_prefix is not None:
            data['parts'].append({
                'type': 'object_pronoun',
                'text': self.object_pronoun_prefix,
                'definition': Object.PRONOUNS[self.object_pronoun_prefix]
            })
        data['parts'].append({
            'type': 'verb_stem',
            'text': self.verb_stem,
            'definition': Verb.TRANSIITIVE_VERBS[self.verb_stem] if self.is_transitive else Verb.INTRANSITIVE_VERBS[self.verb_stem]
        })
        data['parts'].append({
            'type': 'tense',
            'text': self.tense_suffix,
            'definition': self.TENSES[self.tense_suffix]
        })
        return data


class Object:
    SUFFIXES = {
        'eika': 'proximal',
        'oka': 'distal',
        'aa': 'unspecified'
    }
    PRONOUNS = {
        'i': 'me',
        'u': 'him/her/it (distal)',
        'ui': 'them (distal)',
        'ma': 'him/her/it (proximal)',
        'mai': 'them (proximal)',
        'a': 'him/her/it (proximal)',
        'ai': 'them (proximal)',
        'ni': 'us (plural, exclusive)',
        'tei': 'us (plural, inclusive)',
        'ta': 'us (dual), you and I',
        'ü': 'you (singular)',
        'üi': 'you (plural), you all',
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
        if "'" not in self.noun[-2:]: # noun does not end in glottal stop
            if object_suffix == 'eika':
                object_suffix = 'neika'
            elif object_suffix == 'oka':
                object_suffix = 'noka'
            elif object_suffix == 'aa':
                object_suffix = 'na'
        return f"{self.noun}-{object_suffix}"
    
    @classmethod
    def check_agreement(self, object_suffix: str, object_pronoun: str) -> bool:
        """Check whether the object suffix and object pronoun agree
        
        Either both must contain 'proximal', both contain 'distal', or neither contain either.
        """
        if 'proximal' in object_suffix and 'proximal' in object_pronoun:
            return True
        elif 'distal' in object_suffix and 'distal' in object_pronoun:
            return True
        elif 'proximal' not in object_suffix and 'proximal' not in object_pronoun and 'distal' not in object_suffix and 'distal' not in object_pronoun:
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
            return 'aa'
        elif 'proximal' in Object.PRONOUNS[object_pronoun]:
            return 'eika'
        elif 'distal' in Object.PRONOUNS[object_pronoun]:
            return 'oka'
        else:
            return None
        
    @classmethod
    def get_matching_pronouns(self, object_suffix: str) -> List[str]:
        """Get the object pronouns that match the object suffix
        
        If the object suffix is proximal, return the proximal pronouns.
        If the object suffix is distal, return the distal pronouns.
        """
        if object_suffix not in self.SUFFIXES:
            raise ValueError(f"Object suffix must be one of {self.SUFFIXES}")
        elif object_suffix == 'eika':
            return [pronoun for pronoun in self.PRONOUNS if 'proximal' in self.PRONOUNS[pronoun]]
        elif object_suffix == 'oka':
            return [pronoun for pronoun in self.PRONOUNS if 'distal' in self.PRONOUNS[pronoun]]
        elif object_suffix == 'aa':
            return []
            # return [pronoun for pronoun in self.PRONOUNS if 'it' in self.PRONOUNS[pronoun] or 'them' in self.PRONOUNS[pronoun]]
        
    @property
    def details(self) -> Dict:
        data = {
            'type': 'object',
            'text': str(self),
            'parts': []
        }
        data['parts'].append({
            'type': 'noun',
            'text': self.noun,
            'definition': NOUNS[self.noun]
        })
        data['parts'].append({
            'type': 'object_suffix',
            'text': self.object_suffix,
            'definition': self.SUFFIXES[self.object_suffix]
        })
        return data


def get_choice(choices: Iterable[str], prompt: str = "Choose from the following: ") -> str:
    """Get a choice from the user

    Number the choices and print them out. The user inputs the number of their choice.
    
    If the user enters an invalid choice, ask them to try again.
    """
    choices = list(choices)
    while True:
        print(prompt)
        for i, choice in enumerate(choices):
            print(f"{i+1}. {choice}")
        choice = input("Choice: ")
        try:
            choice = int(choice)
        except ValueError:
            print("Please enter a number")
            continue
        if choice < 1 or choice > len(choices):
            print("Please enter a valid choice")
            continue
        return choices[choice-1]
    
def get_random_choice(choices: Iterable[str], *args, **kwargs) -> str:
    """Get a random choice from the given choices"""
    choices = list(choices)
    return random.choice(choices)

# def build_sentence(choice_func: Callable[[Iterable[str]], str] = get_choice) -> Tuple[Subject, Object, Verb]:
#     """Sentences must have a subject, object, and verb in any order.
    
#     Let the user choose what they want to provide next.
#     Depending on the current choices, their options may be restricted.
#     For example, if they have already provided a subject, they cannot provide another subject.
#     If they have provided an object, they must choose a *transitive* verb (and a marching object pronoun).
#     If they have provided an intransitive verb, they cannot provide an object.
#     And so on.
#     """
    
#     subject: Subject = None
#     object: Object = None
#     verb: Verb = None
#     while True:
#         choices = []
#         if subject is None:
#             choices.append('subject')
#         if object is None and (verb is None or (verb.is_transitive and Object.get_matching_suffix(verb.object_pronoun_prefix))):
#             choices.append('object')
#         if verb is None:
#             choices.append('verb')
#         if len(choices) == 0:
#             print("You have provided a subject, object, and verb. You are done!")
#             break
        
#         # choice = get_choice("What would you like to provide next?", choices)
#         choice = choice_func(choices, prompt="What would you like to provide next?")
#         if choice == 'subject':
#             # subject_choice = get_choice("What is the subject?", [*Subject.PRONOUNS.keys(), *NOUNS.keys()])
#             subject_choice = choice_func([*Subject.PRONOUNS.keys(), *NOUNS.keys()], prompt="What is the subject?")
#             if subject_choice in Subject.PRONOUNS:
#                 subject = Subject(subject_choice, subject_suffix=None)
#             else:
#                 # subject_suffix_choice = get_choice("What is the subject suffix?", Subject.SUFFIXES.keys())
#                 subject_suffix_choice = choice_func(Subject.SUFFIXES.keys(), prompt="What is the subject suffix?")
#                 subject = Subject(subject_choice, subject_suffix_choice)
#         elif choice == 'object':
#             if verb is not None and (not verb.is_transitive or Object.get_matching_suffix(verb.object_pronoun_prefix)):
#                 raise ValueError("Cannot provide an object for an intransitive verb")
            
#             # object_choice = get_choice("What is the object?", NOUNS.keys())
#             object_choice = choice_func(NOUNS.keys(), prompt="What is the object?")
#             if verb is not None: # must be transitive
#                 object_suffix_choice = Object.get_matching_suffix(verb.object_pronoun_prefix)
#             else:
#                 # object_suffix_choice = get_choice("What is the object suffix?", Object.SUFFIXES.keys())
#                 object_suffix_choice = choice_func(Object.SUFFIXES.keys(), prompt="What is the object suffix?")
#             object = Object(object_choice, object_suffix_choice)
#         elif choice == 'verb':
#             if object is not None: # verb must be transitive
#                 # verb_choice = get_choice("What is the verb?", Verb.TRANSIITIVE_VERBS.keys())
#                 verb_choice = choice_func(Verb.TRANSIITIVE_VERBS.keys(), prompt="What is the verb?")
#             else:
#                 # verb_choice = get_choice("What is the verb?", [*Verb.TRANSIITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()])
#                 verb_choice = choice_func([*Verb.TRANSIITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()], prompt="What is the verb?")

#             object_pronoun_choice = None
#             if verb_choice in Verb.TRANSIITIVE_VERBS: # transitive - choose object pronoun
#                 if object is not None:
#                     # object_pronoun_choice = get_choice("What is the object pronoun?", Object.get_matching_pronouns(object.object_suffix))
#                     object_pronoun_choice = choice_func(Object.get_matching_pronouns(object.object_suffix), prompt="What is the object pronoun?")
#                 else:
#                     # object_pronoun_choice = get_choice("What is the object pronoun?", Object.PRONOUNS.keys())
#                     object_pronoun_choice = choice_func(Object.PRONOUNS.keys(), prompt="What is the object pronoun?")
                
#             # get verb tense
#             # verb_tense_choice = get_choice("What is the verb tense?", Verb.TENSES.keys())
#             verb_tense_choice = choice_func(Verb.TENSES.keys(), prompt="What is the verb tense?")

#             verb = Verb(verb_choice, verb_tense_choice, object_pronoun_choice)

#     return subject, verb, object

def print_sentence(subject: Subject, verb: Verb, object: Object = None):
    if subject.noun in Subject.PRONOUNS:
        if object:
            print(f"{object} {subject} {verb}")
        else:
            print(f"{subject} {verb}")
    else:
        if object:
            print(f"{object} {verb} {subject}")
        else:
            print(f"{verb} {subject}")

def get_all_choices(subject_noun: Optional[str],
                    subject_suffix: Optional[str],
                    verb: Optional[str],
                    verb_tense: Optional[str],
                    object_pronoun: Optional[str],
                    object_noun: Optional[str],
                    object_suffix: Optional[str]) -> Dict[str, Any]:
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
        if object_pronoun not in Object.get_matching_pronouns(object_suffix):
            object_suffix = None

    # Subject
    choices['subject_noun'] = {
        'choices': [*Subject.PRONOUNS.keys(), *NOUNS.keys()],
        'value': subject_noun,
        'requirement': "required"
    }
    if subject_noun in Subject.PRONOUNS:
        choices['subject_suffix'] = {
            'choices': [],
            'value': None,
            'requirement': "disabled"
        }
    else:
        choices['subject_suffix'] = {
            'choices': list(Subject.SUFFIXES.keys()),
            'value': subject_suffix,
            'requirement': "required"
        }

    # Verb
    if object_noun is not None: # verb must be transitive
        choices['verb'] = {
            'choices': list(Verb.TRANSIITIVE_VERBS.keys()),
            'value': verb,
            'requirement': "required"
        }
    else:
        choices['verb'] = {
            'choices': [*Verb.TRANSIITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()],
            'value': verb,
            'requirement': "required"
        }

    # Verb tense
    if verb is None:
        choices['verb_tense'] = {
            'choices': [],
            'value': None,
            'requirement': "disabled"
        }
    else:
        choices['verb_tense'] = {
            'choices': list(Verb.TENSES.keys()),
            'value': verb_tense,
            'requirement': "required"
        }

    # Object pronoun
    if verb is None or verb in Verb.INTRANSITIVE_VERBS: 
        choices['object_pronoun'] = {
            'choices': [],
            'value': None,
            'requirement': "disabled"
        }
    elif object_suffix is not None: # object pronoun must match object suffix
        choices['object_pronoun'] = {
            'choices': Object.get_matching_pronouns(object_suffix),
            'value': object_pronoun,
            'requirement': "required"
        }
    else:
        choices['object_pronoun'] = {
            'choices': list(Object.PRONOUNS.keys()),
            'value': object_pronoun,
            'requirement': "optional"
        }
    

    # Object noun
    # if verb is intransitive, object noun must be None
    if verb in Verb.INTRANSITIVE_VERBS:
        choices['object_noun'] = {
            'choices': [],
            'value': None,
            'requirement': "disabled"
        }
    else: # verb is not selected or is transitive
        choices['object_noun'] = {
            'choices': list(NOUNS.keys()),
            'value': object_noun,
            'requirement': "required"
        }

    # Object suffix
    if object_noun is None:
        choices['object_suffix'] = {
            'choices': [],
            'value': None,
            'requirement': "disabled"
        }
    elif object_pronoun is not None:
        print("Object pronoun is not None", object_pronoun, Object.get_matching_suffix(object_pronoun))
        choices['object_suffix'] = {
            'choices': [] if Object.get_matching_suffix(object_pronoun) is None else [Object.get_matching_suffix(object_pronoun)],
            'value': Object.get_matching_suffix(object_pronoun) or "",
            'requirement': "required"
        }
    else:
        choices['object_suffix'] = {
            'choices': list(Object.SUFFIXES.keys()),
            'value': object_suffix,
            'requirement': "required"
        }

    return choices

def format_sentence(subject_noun: Optional[str],
                    subject_suffix: Optional[str],
                    verb_stem: Optional[str],
                    verb_tense: Optional[str],
                    object_pronoun: Optional[str],
                    object_noun: Optional[str],
                    object_suffix: Optional[str]) -> List[Dict]:
    subject = Subject(subject_noun, subject_suffix)
    verb = Verb(verb_stem, verb_tense, object_pronoun)
    object = None
    try:
        object = Object(object_noun, object_suffix)
    except ValueError as e:
        print(e)
    
    if subject.noun in Subject.PRONOUNS:
        if object:
            return [object.details, subject.details, verb.details]
        else:
            return [subject.details, verb.details]
    else:
        if object:
            return [object.details, verb.details, subject.details]
        else:
             return [verb.details, subject.details]
        
def main():
    for i in range(1000):
        # subject, verb, object = build_sentence(choice_func=get_random_choice)
        # print_sentence(subject, verb, object)
        pass

if __name__ == '__main__':
    main()




    