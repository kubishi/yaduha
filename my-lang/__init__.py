"""
My Language Language Definition

A test constructed language
"""

from typing import List, Tuple, Type
from pydantic import BaseModel, Field
from yaduha.language import Sentence, VocabEntry

# Import vocabulary
from my-lang.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

# Language metadata
LANGUAGE_NAME = "My Language"
LANGUAGE_CODE = "myl"
LANGUAGE_DESCRIPTION = """A test constructed language"""


# ============================================================================
# SENTENCE TYPES
# Define your sentence structures here
# ============================================================================

class SimpleSentence(Sentence["SimpleSentence"]):
    """
    A basic SOV sentence.

    Example: "The cat sleeps."
    """
    subject: str = Field(..., description="The subject of the sentence")
    verb: str = Field(..., description="The verb")

    def __str__(self) -> str:
        # TODO: Implement rendering to My Language
        return f"{self.subject} {self.verb}"

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SimpleSentence"]]:
        return [
            ("The cat sleeps.", SimpleSentence(subject="cat", verb="sleep")),
        ]


# Export sentence types
SENTENCE_TYPES = (SimpleSentence,)
