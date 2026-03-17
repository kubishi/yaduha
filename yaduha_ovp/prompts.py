from typing import Iterable, List, Type, TYPE_CHECKING
from yaduha_ovp.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS
from yaduha_ovp import (
    LENIS_MAP, SubjectVerbSentence, SubjectVerbObjectSentence
)

if TYPE_CHECKING:
    from yaduha.language import Sentence

SYSTEM_PROMPT_PREFIX = (
    "You are a translator that translates English sentences into Owens Valley Paiute. "
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible. "
    "It doesn't need to be perfect and you can leave English words untranslated if necessary.\n"
)

TOOL_USE_INSTRUCTION = (
    "You may also have access to tools that can help you produce a better translation. "
    "Use these tools as needed. You can make one or many tool calls (in parallel and/or sequentially) "
    "until you decide to respond.\n"
)

VOCABULARY_PROMPT = (
    "You use the following vocabulary to translate user input sentences from English to Owens Valley Paiute.\n" +
    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible.\n" +
    "It doesn't need to be perfect and you can leave English words untranslated if necessary.\n" +
    "# Vocabulary\n" +
    "## Nouns: \n" + "\n".join([f"{noun.target}: {noun.english}" for noun in NOUNS]) + "\n" +
    "## Transitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in TRANSITIVE_VERBS]) + "\n" +
    "## Intransitive Verbs: \n" + "\n".join([f"{verb.target}: {verb.english}" for verb in INTRANSITIVE_VERBS]) + "\n"
)

SENTENCE_STRUCTURE_PROMPT = (
    "# Sentence Structure\n" +
    "## Simple Sentence Structure: \n" +
    "Subject-Object-Verb: [object noun]-[object suffix] [subject noun]-[subject suffix] [object pronoun]-[verb]-[verb tense]\n" +
    "Subject Pronoun-Object-Verb: [object noun]-[object suffix] [subject pronoun] [object pronoun]-[verb]-[verb tense]\n" +
    "Subject-Verb: [verb]-[verb tense] [subject noun]-[subject suffix]\n" +
    "## Verb Nominalization Sentence Structure: \n" +
    "Subject Nominalizer: [verb]-[verb nominalizer tense]-[subject suffix] [verb nominalizer]-[verb nominalizer tense]\n" +
    "Object Nominalizer: [verb]-[verb nominalizer tense]-[object suffix] [subject noun]-[subject suffix] [object pronoun]-[verb]-[verb tense]\n" +
    "Subject&Object Nominalizer: [verb]-[verb nominalizer tense]-[object suffix] [verb nominalizer]-[verb nominalizer tense]-[subject suffix] [subject noun]-[subject suffix] [object pronoun]-[verb]-[verb tense]\n"
)

FORTIS_LENIS_PROMPT = (
    "# Fortis/Lenis Transformations\n" +
    ", ".join([f"{f}->{l}" for f, l in LENIS_MAP.items()]) + "\n"
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
    system_prompt += FORTIS_LENIS_PROMPT
    for sentence_cls in include_examples:
        for source, example_sentence in sentence_cls.get_examples():
            system_prompt += (
                "\n# Example\n"
                f"English: {source}\n"
                f"Owens Valley Paiute: {example_sentence}\n"
            )

    return system_prompt
