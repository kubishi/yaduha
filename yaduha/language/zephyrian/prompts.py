from typing import Iterable, List, Type, TYPE_CHECKING
from yaduha.language.zephyrian.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS
from yaduha.language.zephyrian import (
    SubjectVerbSentence, SubjectVerbObjectSentence,
    Evidentiality, Tense, Aspect, Animacy
)

if TYPE_CHECKING:
    from yaduha.language import Sentence

SYSTEM_PROMPT_PREFIX = (
    "You are a translator that translates English sentences into Zephyrian. "
    "Zephyrian is a VSO (Verb-Subject-Object) language with evidentiality marking, "
    "vowel harmony, and an animacy distinction. "
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible. "
    "It doesn't need to be perfect and you can leave English words untranslated if necessary.\n"
)

TOOL_USE_INSTRUCTION = (
    "You may also have access to tools that can help you produce a better translation. "
    "Use these tools as needed. You can make one or many tool calls (in parallel and/or sequentially) "
    "until you decide to respond.\n"
)

VOCABULARY_PROMPT = (
    "You use the following vocabulary to translate user input sentences from English to Zephyrian.\n"
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible.\n"
    "It doesn't need to be perfect and you can leave English words untranslated if necessary.\n"
    "# Vocabulary\n"
    "## Nouns: \n" + "\n".join([f"{noun.target}: {noun.english}" for noun in NOUNS]) + "\n"
    "## Transitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in TRANSITIVE_VERBS]) + "\n"
    "## Intransitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in INTRANSITIVE_VERBS]) + "\n"
)

SENTENCE_STRUCTURE_PROMPT = (
    "# Sentence Structure\n"
    "Zephyrian uses VSO (Verb-Subject-Object) word order.\n"
    "Zephyrian is agglutinative - morphemes are concatenated without separators.\n\n"
    "## Verb Conjugation Structure:\n"
    "[evidential][aspect-modified stem][tense] (all joined together)\n"
    "Example: 'vasōmā' = va + sōm + ā = witnessed + sleep + present\n\n"
    "## Evidential Prefixes (how you know the information):\n"
    "- va : witnessed (you saw it directly)\n"
    "- shi : reported (you heard it from someone)\n"
    "- zo : inferred (you deduced it from evidence)\n\n"
    "## Aspect Infixes (inserted after first consonant):\n"
    "- (none) : simple action\n"
    "- el : continuous action\n"
    "- or : perfective (completed) action\n\n"
    "## Tense Suffixes (with vowel harmony):\n"
    "- Past: àn (back vowels) / èn (front vowels)\n"
    "- Present: ā (back vowels) / ē (front vowels)\n"
    "- Future: úr (back vowels) / ír (front vowels)\n\n"
    "## Noun Structure:\n"
    "[article][noun][plural] (all joined together)\n"
    "Example: 'elzāfir' = el + zāfir = definite.animate + wolf\n\n"
    "## Articles (based on animacy and definiteness):\n"
    "- el : definite animate\n"
    "- ul : definite inanimate\n"
    "- en : indefinite animate\n"
    "- un : indefinite inanimate\n\n"
    "## Plural Suffixes (vowel harmony):\n"
    "- ath : for words with back vowels (a, o, u)\n"
    "- ith : for words with front vowels (e, i, y)\n"
    "Example: 'elzāfirath' = the wolves (animate, plural)\n\n"
    "## Object Marking:\n"
    "Objects take the accusative prefix 'ko' before the article.\n"
    "Example: 'koelzāfir' = ko + el + zāfir = ACC + the + wolf\n\n"
    "## Pronouns:\n"
    "Subject: zē (I), zēn (we), thū (you.sg), thūn (you.pl), vā (he/she/it), vān (they)\n"
    "Object: mē (me), mēn (us), thē (you.sg), thēn (you.pl), lā (him/her/it), lān (them)\n"
)

VOWEL_HARMONY_PROMPT = (
    "# Vowel Harmony Rules\n"
    "Zephyrian uses vowel harmony. Suffixes change based on the vowel class of the word:\n"
    "- Front vowels: e, i, y, ē, ī\n"
    "- Back vowels: a, o, u, ā, ō, ū\n"
    "The first vowel in the stem determines which suffix variant to use.\n"
)


def get_prompt(include_vocab: bool,
               has_tools: bool = False,
               include_examples: Iterable[Type["Sentence"]] | None = None) -> str:
    include_examples = include_examples or []
    system_prompt = SYSTEM_PROMPT_PREFIX
    if has_tools:
        system_prompt += TOOL_USE_INSTRUCTION
    if include_vocab:
        system_prompt += VOCABULARY_PROMPT
    system_prompt += SENTENCE_STRUCTURE_PROMPT
    system_prompt += VOWEL_HARMONY_PROMPT

    for sentence_cls in include_examples:
        for source, example_sentence in sentence_cls.get_examples():
            system_prompt += (
                "\n# Example\n"
                f"English: {source}\n"
                f"Zephyrian: {example_sentence}\n"
            )

    return system_prompt
