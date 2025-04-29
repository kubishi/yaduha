from yaduha.syntax import (
    SentenceList, Sentence,
    Pronoun, Verb, SubjectNoun, ObjectNoun,
    Person, Plurality, Proximity, Inclusivity,
    Tense, Aspect
)
from pydantic import ValidationError

try:
    EXAMPLE_SENTENCES = [
        {
            "sentence": "I am sitting in a chair.",
            "simple": "I am sitting.",
            "comparator": "I am sitting.",
            "translation": "Nüü katünnu-ba' katü-ti.",
            "translation_simple": "Katü-ti nüü.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=Pronoun(
                            person=Person.first,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="sit",
                            tense=Tense.present,
                            aspect=Aspect.continuous
                        ),
                        object=None
                    )
                ]
            )
        },
        {
            "sentence": "The one who ran has seen Rebecca and met her.",
            "simple": "The one who ran has seen Rebecca. He met her.",
            "comparator": "The one who ran has seen [OBJECT]. He [VERB] her.",
            "translation": "Poyoha-dü-uu Rebecca-noka u-buni-ku-si u-[met]-ku.",
            "translation_simple": "Poyoha-dü-uu Rebecca-noka u-buni-pü. Mahu u-[met]-ku.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head=Verb(
                                lemma="run",
                                tense=Tense.past,
                                aspect=Aspect.completive
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular  
                        ),
                        verb=Verb(
                            lemma="see",
                            tense=Tense.present,
                            aspect=Aspect.perfect
                        ),
                        object=ObjectNoun(
                            head="Rebecca",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    ),
                    Sentence(
                        subject=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="meet",
                            tense=Tense.past,
                            aspect=Aspect.simple
                        ),
                        object=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=True
                        ),
                    )
                ]
            )
        },
        {
            "sentence": "The dogs were chasing their tails.",
            "simple": "The dogs were chasing their tails.",
            "comparator": "The dogs were chasing their tails.",
            "translation": "Isha'pugu-ii tüi-kwadzi-neika a-maŵia-ti.",
            "translation_simple": "Isha'pugu-ii tüi-kwadzi-neika a-maŵia-ti.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head="dog",
                            proximity=Proximity.proximal,
                            plurality=Plurality.plural
                        ),
                        verb=Verb(
                            lemma="chase",
                            tense=Tense.past,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="tail",
                            possessive_determiner=Pronoun(
                                person=Person.third,
                                plurality=Plurality.plural,
                                proximity=Proximity.proximal,
                                inclusivity=Inclusivity.exclusive,
                                reflexive=True
                            ),
                            proximity=Proximity.proximal,
                            plurality=Plurality.plural
                        )
                    )
                ]
            )
        },
        {
            "sentence": "The fighter is eating his street.",
            "simple": "The fighter is eating his street.",
            "comparator": "The [VERB]-er is eating his [OBJECT].",
            "translation": "Nappidügü-dü-ii tü-boyo-neika a-düka-ti.",
            "translation_simple": "Nappidügü-dü-ii tü-[street]-neika a-düka-ti.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head=Verb(
                                lemma="fight",
                                tense=Tense.present,
                                aspect=Aspect.simple
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="eat",
                            tense=Tense.present,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="street",
                            possessive_determiner=Pronoun(
                                person=Person.first,
                                plurality=Plurality.singular,
                                proximity=Proximity.proximal,
                                inclusivity=Inclusivity.exclusive,
                                reflexive=True
                            ),
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    )
                ]
            )
        },
        {
            "sentence": "The book sits on the table.",
            "simple": "The book sits.",
            "comparator": "The [SUBJECT] sits.",
            "translation": "[book]-uu tibo-ba' katü-ti.",
            "translation_simple": "[book]-uu katü-ti.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head="book",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="sit",
                            tense=Tense.present,
                            aspect=Aspect.simple
                        ),
                        object=None
                    )
                ]
            )
        },
        {
            "sentence": "The boy saw it.",
            "simple": "The boy saw it.",
            "comparator": "The [SUBJECT] saw it.",
            "translation": "Naatsi'-ii a-buni-ku.",
            "translation_simple": "Naatsi'-ii a-buni-ku.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head="boy",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="see",
                            tense=Tense.past,
                            aspect=Aspect.completive
                        ),
                        object=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.distal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        )
                    )
                ]
            )
        },
        {
            "sentence": "I saw two men walking their dogs yesterday at Starbucks while drinking a cup of coffee",
            "simple": "I saw men. They were walking. Their dogs were walking. They were drinking coffee.",
            "comparator": "I saw [OBJECT]. They were walking. Their dogs were walking. They were drinking coffee.",
            "translation": "Yongo' nüü waha-ggu naana-noka u-buni-ti Starbucks-wae, tüi-isha'pugu-hodokka hukaŵia-ni tüi-koopi'neika ui-hibi-nna.",
            "translation_simple": "Yongo' nüü naana-noka u-buni-ti. Uhuŵa hukaŵia-ti. Ui-isha'pugu-uu hukaŵia-ti. Uhuŵa koopi'neika u-hibi-ti.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=Pronoun(
                            person=Person.first,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="see",
                            tense=Tense.past,
                            aspect=Aspect.simple
                        ),
                        object=ObjectNoun(
                            head="man",
                            proximity=Proximity.distal,
                            plurality=Plurality.dual
                        )
                    ),
                    Sentence(
                        subject=Pronoun(
                            person=Person.third,
                            plurality=Plurality.dual,
                            proximity=Proximity.distal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="walk",
                            tense=Tense.past,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="dog",
                            possessive_determiner=Pronoun(
                                person=Person.third,
                                plurality=Plurality.dual,
                                proximity=Proximity.proximal,
                                inclusivity=Inclusivity.exclusive,
                                reflexive=True
                            ),
                            proximity=Proximity.proximal,
                            plurality=Plurality.plural
                        )
                    ),
                    Sentence(
                        subject=Pronoun(
                            person=Person.third,
                            plurality=Plurality.singular,
                            proximity=Proximity.proximal,
                            inclusivity=Inclusivity.exclusive,
                            reflexive=False
                        ),
                        verb=Verb(
                            lemma="drink",
                            tense=Tense.past,
                            aspect=Aspect.continuous
                        ),
                        object=ObjectNoun(
                            head="coffee",
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    )
                ]
            )
        },
        {
            "sentence": "That runner down the street will eat the one that has fallen.",
            "simple": "That runner will eat the one that has fallen.",
            "comparator": "That runner will eat the one that has fallen.",
            "translation": "Poyoha-dü-ii kwatsa'i-doka u-düka-wei.",
            "translation_simple": "Poyoha-dü-ii kwatsa'i-doka u-düka-wei.",
            "response": SentenceList(
                sentences=[
                    Sentence(
                        subject=SubjectNoun(
                            head=Verb(
                                lemma="run",
                                tense=Tense.present,
                                aspect=Aspect.simple
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.distal,
                            plurality=Plurality.singular
                        ),
                        verb=Verb(
                            lemma="eat",
                            tense=Tense.future,
                            aspect=Aspect.simple
                        ),
                        object=ObjectNoun(
                            head=Verb(
                                lemma="fall",
                                tense=Tense.present,
                                aspect=Aspect.perfect
                            ),
                            possessive_determiner=None,
                            proximity=Proximity.proximal,
                            plurality=Plurality.singular
                        )
                    )
                ]
            )
        },
    ]
except ValidationError as exc:
    print(exc.errors())
    raise exc