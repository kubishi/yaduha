"""
Vocabulary Assistant

Helps users add vocabulary to their language with suggestions for morphology patterns,
related words, and phonological consistency.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

from yaduha.agent import Agent
from yaduha.language import VocabEntry, LoadedLanguage

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class WordCategory(str, Enum):
    """Categories of words that can be added to vocabulary."""
    noun = "noun"
    transitive_verb = "transitive_verb"
    intransitive_verb = "intransitive_verb"
    adjective = "adjective"
    adverb = "adverb"
    pronoun = "pronoun"
    conjunction = "conjunction"
    preposition = "preposition"
    interjection = "interjection"


class MorphologyPattern(BaseModel):
    """A morphological pattern detected or suggested for the language."""
    name: str = Field(..., description="Name of the pattern (e.g., 'vowel harmony', 'consonant mutation')")
    description: str = Field(..., description="Description of how the pattern works")
    examples: List[str] = Field(default_factory=list, description="Examples of the pattern in existing vocabulary")


class VocabularySuggestion(BaseModel):
    """A suggested vocabulary entry with rationale."""
    english: str = Field(..., description="English word or phrase")
    target: str = Field(..., description="Suggested word in the target language")
    category: WordCategory = Field(..., description="Grammatical category")
    rationale: str = Field(..., description="Why this form was suggested (phonology, morphology, etc.)")
    alternatives: List[str] = Field(default_factory=list, description="Alternative target language forms")
    related_words: List[str] = Field(default_factory=list, description="Related words that might also be added")
    morphology_notes: Optional[str] = Field(default=None, description="Notes about morphological patterns used")


class VocabularySuggestions(BaseModel):
    """Container for multiple vocabulary suggestions."""
    suggestions: List[VocabularySuggestion] = Field(..., description="List of vocabulary suggestions")
    detected_patterns: List[MorphologyPattern] = Field(
        default_factory=list,
        description="Morphological patterns detected in existing vocabulary"
    )


class _PatternAnalysis(BaseModel):
    """Internal model for pattern analysis response."""
    patterns: List[MorphologyPattern] = Field(..., description="Detected patterns")


class VocabularyAssistant:
    """
    LLM-powered assistant for adding and managing vocabulary.

    Analyzes existing vocabulary to detect patterns and suggests new words
    that fit the phonological and morphological patterns of the language.
    """

    def __init__(self, agent: Agent[Any], language: Optional[LoadedLanguage] = None):
        """
        Initialize the vocabulary assistant.

        Args:
            agent: The LLM agent to use for suggestions
            language: Optional loaded language to analyze for patterns
        """
        self.agent = agent
        self.language = language
        self._detected_patterns: Optional[List[MorphologyPattern]] = None

    def _get_existing_vocab_summary(self) -> str:
        """Get a summary of existing vocabulary for context."""
        if not self.language:
            return "No existing vocabulary loaded."

        lines = [f"# Existing Vocabulary for {self.language.name} ({self.language.code})"]

        if self.language.nouns:
            lines.append("\n## Nouns")
            for entry in self.language.nouns[:20]:  # Limit for prompt size
                lines.append(f"- {entry.english} → {entry.target}")

        if self.language.transitive_verbs:
            lines.append("\n## Transitive Verbs")
            for entry in self.language.transitive_verbs[:20]:
                lines.append(f"- {entry.english} → {entry.target}")

        if self.language.intransitive_verbs:
            lines.append("\n## Intransitive Verbs")
            for entry in self.language.intransitive_verbs[:20]:
                lines.append(f"- {entry.english} → {entry.target}")

        if self.language.adjectives:
            lines.append("\n## Adjectives")
            for entry in self.language.adjectives[:20]:
                lines.append(f"- {entry.english} → {entry.target}")

        if self.language.adverbs:
            lines.append("\n## Adverbs")
            for entry in self.language.adverbs[:20]:
                lines.append(f"- {entry.english} → {entry.target}")

        return "\n".join(lines)

    def analyze_patterns(self) -> List[MorphologyPattern]:
        """
        Analyze existing vocabulary to detect morphological and phonological patterns.

        Returns:
            List of detected patterns
        """
        if self._detected_patterns is not None:
            return self._detected_patterns

        system_prompt = """You are a linguistic analyst specializing in morphology and phonology.
Analyze the given vocabulary and identify patterns such as:
- Vowel harmony (front/back vowels, high/low vowels)
- Consonant mutations (lenition, fortition)
- Affixation patterns (prefixes, suffixes, infixes)
- Syllable structure (CV, CVC, etc.)
- Phonotactic constraints
- Semantic categorization patterns (e.g., animals start with similar sounds)

Be specific and provide examples from the vocabulary."""

        vocab_summary = self._get_existing_vocab_summary()

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this vocabulary for patterns:\n\n{vocab_summary}"}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=_PatternAnalysis
        )

        self._detected_patterns = response.content.patterns
        return self._detected_patterns

    def suggest_word(
        self,
        english_word: str,
        category: WordCategory,
        context: Optional[str] = None,
        style_hints: Optional[List[str]] = None,
    ) -> VocabularySuggestion:
        """
        Suggest a target language word for a given English word.

        Args:
            english_word: The English word to translate
            category: Grammatical category of the word
            context: Optional context for how the word will be used
            style_hints: Optional hints about desired style (e.g., "short", "formal")

        Returns:
            A vocabulary suggestion with rationale
        """
        system_prompt = """You are a constructed language (conlang) specialist helping design vocabulary.
Your task is to suggest a word in the target language that:
1. Fits the phonological patterns of the existing vocabulary
2. Follows any morphological rules you detect
3. Is memorable and pronounceable
4. Makes sense for the semantic category

Provide your reasoning and suggest alternatives."""

        vocab_summary = self._get_existing_vocab_summary()

        user_content = f"""Suggest a word for:
- English: {english_word}
- Category: {category.value}
{f'- Context: {context}' if context else ''}
{f'- Style hints: {", ".join(style_hints)}' if style_hints else ''}

Existing vocabulary for reference:
{vocab_summary}"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=VocabularySuggestion
        )

        return response.content

    def suggest_batch(
        self,
        words: List[tuple[str, WordCategory]],
        maintain_consistency: bool = True,
    ) -> VocabularySuggestions:
        """
        Suggest target language words for multiple English words.

        Args:
            words: List of (english_word, category) tuples
            maintain_consistency: If True, ensure suggestions are consistent with each other

        Returns:
            Container with all suggestions and detected patterns
        """
        _ = maintain_consistency  # Used in prompt context

        system_prompt = """You are a constructed language (conlang) specialist helping design vocabulary.
Your task is to suggest words in the target language that:
1. Fit the phonological patterns of the existing vocabulary
2. Follow any morphological rules you detect
3. Are internally consistent with each other
4. Are memorable and pronounceable

For each word, explain your reasoning. Also identify any patterns in the existing vocabulary."""

        vocab_summary = self._get_existing_vocab_summary()

        word_list = "\n".join(f"- {word} ({cat.value})" for word, cat in words)

        user_content = f"""Suggest words for the following:
{word_list}

Existing vocabulary for reference:
{vocab_summary}

Please ensure the new words are consistent with each other and with the existing vocabulary."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=VocabularySuggestions
        )

        return response.content

    def suggest_related_words(
        self,
        base_word: VocabEntry,
        relationship_types: Optional[List[str]] = None,
    ) -> VocabularySuggestions:
        """
        Suggest related words based on an existing vocabulary entry.

        Args:
            base_word: The existing word to base suggestions on
            relationship_types: Types of relationships (e.g., "antonym", "diminutive", "agent noun")

        Returns:
            Container with related word suggestions
        """
        if relationship_types is None:
            relationship_types = ["antonym", "related action", "agent noun", "derived adjective"]

        system_prompt = """You are a constructed language (conlang) specialist helping expand vocabulary.
Given a base word, suggest related words that follow morphological derivation patterns.
Consider:
- Semantic relationships (antonyms, hyponyms, meronyms)
- Morphological derivations (agent nouns, diminutives, augmentatives)
- Related concepts in the same semantic field

Ensure derived words follow consistent morphological patterns."""

        vocab_summary = self._get_existing_vocab_summary()

        user_content = f"""Base word:
- English: {base_word.english}
- Target: {base_word.target}

Suggest related words with these relationships: {", ".join(relationship_types)}

Existing vocabulary for reference:
{vocab_summary}"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=VocabularySuggestions
        )

        return response.content

    def generate_vocab_code(self, suggestions: List[VocabularySuggestion]) -> str:
        """
        Generate Python code for adding vocabulary entries.

        Args:
            suggestions: List of vocabulary suggestions to convert to code

        Returns:
            Python code string for vocab.py
        """
        lines = ["from yaduha.language import VocabEntry", ""]

        # Group by category
        by_category: Dict[WordCategory, List[VocabularySuggestion]] = {}
        for s in suggestions:
            if s.category not in by_category:
                by_category[s.category] = []
            by_category[s.category].append(s)

        category_to_var = {
            WordCategory.noun: "NOUNS",
            WordCategory.transitive_verb: "TRANSITIVE_VERBS",
            WordCategory.intransitive_verb: "INTRANSITIVE_VERBS",
            WordCategory.adjective: "ADJECTIVES",
            WordCategory.adverb: "ADVERBS",
        }

        for category, items in by_category.items():
            var_name = category_to_var.get(category, f"{category.value.upper()}S")
            lines.append(f"# {category.value.replace('_', ' ').title()}s")
            lines.append(f"{var_name} = [")
            for item in items:
                lines.append(f'    VocabEntry(english="{item.english}", target="{item.target}"),')
            lines.append("]")
            lines.append("")

        return "\n".join(lines)
