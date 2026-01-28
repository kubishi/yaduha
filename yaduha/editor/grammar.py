"""
Grammar Helper

Guides users through defining new sentence types, grammatical features,
and morphological rules for their constructed language.
"""

from typing import List, Optional, Dict, Any, Type, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

from yaduha.agent import Agent
from yaduha.language import Sentence, LoadedLanguage

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class WordOrder(str, Enum):
    """Common word order typologies."""
    svo = "SVO"  # Subject-Verb-Object (English)
    sov = "SOV"  # Subject-Object-Verb (Japanese)
    vso = "VSO"  # Verb-Subject-Object (Welsh)
    vos = "VOS"  # Verb-Object-Subject (Malagasy)
    osv = "OSV"  # Object-Subject-Verb (rare)
    ovs = "OVS"  # Object-Verb-Subject (rare)
    free = "free"  # Free word order with case marking


class GrammaticalFeature(str, Enum):
    """Common grammatical features that can be added to a language."""
    tense = "tense"
    aspect = "aspect"
    mood = "mood"
    evidentiality = "evidentiality"
    person = "person"
    number = "number"
    gender = "gender"
    case = "case"
    definiteness = "definiteness"
    animacy = "animacy"
    honorifics = "honorifics"
    voice = "voice"
    polarity = "polarity"


class GrammarFeature(BaseModel):
    """Definition of a grammatical feature for the language."""
    name: str = Field(..., description="Name of the feature (e.g., 'Tense')")
    feature_type: GrammaticalFeature = Field(..., description="Type of grammatical feature")
    description: str = Field(..., description="What this feature expresses")
    values: List[str] = Field(..., description="Possible values (e.g., ['past', 'present', 'future'])")
    marking_strategy: str = Field(
        ...,
        description="How this feature is marked (prefix, suffix, infix, separate word, tone, etc.)"
    )
    markers: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of values to their morphological markers"
    )
    applies_to: List[str] = Field(
        default_factory=list,
        description="Word classes this feature applies to (e.g., ['verb', 'noun'])"
    )
    examples: List[str] = Field(default_factory=list, description="Example sentences showing the feature")


class SentenceTypeTemplate(BaseModel):
    """Template for a new sentence type."""
    name: str = Field(..., description="Name of the sentence type (e.g., 'SubjectVerbObjectSentence')")
    description: str = Field(..., description="What this sentence type expresses")
    word_order: WordOrder = Field(..., description="Word order for this sentence type")
    required_components: List[str] = Field(
        ...,
        description="Required components (e.g., ['subject', 'verb', 'object'])"
    )
    optional_components: List[str] = Field(
        default_factory=list,
        description="Optional components (e.g., ['adverb', 'adjective'])"
    )
    features: List[GrammarFeature] = Field(
        default_factory=list,
        description="Grammatical features used in this sentence type"
    )
    rendering_rules: List[str] = Field(
        default_factory=list,
        description="Rules for rendering the sentence to target language"
    )
    pydantic_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of field names to Pydantic types for code generation"
    )
    example_english: List[str] = Field(
        default_factory=list,
        description="Example English sentences this type can express"
    )


class SentenceTypeAnalysis(BaseModel):
    """Analysis of existing sentence types."""
    existing_types: List[str] = Field(..., description="Names of existing sentence types")
    covered_patterns: List[str] = Field(..., description="Grammatical patterns already covered")
    gaps: List[str] = Field(..., description="Common patterns not yet covered")
    suggestions: List[str] = Field(..., description="Suggested sentence types to add")


class GrammarDesign(BaseModel):
    """Complete grammar design for a sentence type."""
    template: SentenceTypeTemplate = Field(..., description="The sentence type template")
    python_code: str = Field(..., description="Generated Python code for the sentence type")
    test_sentences: List[tuple[str, str]] = Field(
        default_factory=list,
        description="Test cases as (english, expected_output) pairs"
    )


class GrammarHelper:
    """
    LLM-powered assistant for designing grammar and sentence types.

    Helps users:
    - Analyze existing grammar coverage
    - Design new sentence types
    - Define grammatical features
    - Generate Python code for implementations
    """

    def __init__(self, agent: Agent[Any], language: Optional[LoadedLanguage] = None):
        """
        Initialize the grammar helper.

        Args:
            agent: The LLM agent to use for suggestions
            language: Optional loaded language to analyze
        """
        self.agent = agent
        self.language = language

    def _get_existing_grammar_summary(self) -> str:
        """Get a summary of existing grammar for context."""
        if not self.language:
            return "No existing grammar loaded."

        lines = [f"# Existing Grammar for {self.language.name} ({self.language.code})"]

        if self.language.description:
            lines.append(f"\n{self.language.description}")

        lines.append("\n## Sentence Types")
        for st in self.language.sentence_types:
            lines.append(f"\n### {st.__name__}")
            if st.__doc__:
                lines.append(st.__doc__.strip())

            # Get examples
            try:
                examples = st.get_examples()
                lines.append("\nExamples:")
                for eng, sentence in examples[:3]:
                    lines.append(f"- \"{eng}\" → \"{str(sentence)}\"")
            except Exception:
                pass

            # Get model fields if it's a Pydantic model
            if hasattr(st, 'model_fields'):
                lines.append("\nFields:")
                for field_name, field_info in st.model_fields.items():
                    ann = field_info.annotation
                    lines.append(f"- {field_name}: {ann}")

        return "\n".join(lines)

    def analyze_grammar(self) -> SentenceTypeAnalysis:
        """
        Analyze existing sentence types and identify gaps.

        Returns:
            Analysis of grammar coverage with suggestions
        """
        system_prompt = """You are a linguistic analyst specializing in grammar and syntax.
Analyze the given sentence types and identify:
1. What grammatical patterns are already covered
2. What common patterns are missing
3. Suggestions for new sentence types to add

Consider patterns like:
- Questions (yes/no, wh-questions)
- Negation
- Relative clauses
- Conditional sentences
- Passive voice
- Imperative/commands
- Copular sentences (X is Y)
- Existential sentences (There is X)
- Possessive constructions"""

        grammar_summary = self._get_existing_grammar_summary()

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this grammar:\n\n{grammar_summary}"}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=SentenceTypeAnalysis
        )

        return response.content

    def suggest_sentence_type(
        self,
        pattern_description: str,
        example_sentences: Optional[List[str]] = None,
        word_order: Optional[WordOrder] = None,
    ) -> SentenceTypeTemplate:
        """
        Get suggestions for implementing a new sentence type.

        Args:
            pattern_description: Description of what the sentence type should express
            example_sentences: Optional example English sentences
            word_order: Optional word order to use (defaults to analyzing existing types)

        Returns:
            Template for the new sentence type
        """
        system_prompt = """You are a constructed language (conlang) specialist helping design grammar.
Your task is to design a new sentence type that:
1. Fits the existing grammar patterns of the language
2. Handles the described grammatical pattern
3. Is implementable in Python using Pydantic models

Provide:
- A clear structure with required/optional components
- Grammatical features that apply
- Rules for rendering to the target language
- Pydantic field definitions for code generation"""

        grammar_summary = self._get_existing_grammar_summary()

        user_content = f"""Design a sentence type for:
{pattern_description}

{f'Example sentences: {example_sentences}' if example_sentences else ''}
{f'Word order: {word_order.value}' if word_order else ''}

Existing grammar for reference:
{grammar_summary}"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=SentenceTypeTemplate
        )

        return response.content

    def design_grammar_feature(
        self,
        feature_type: GrammaticalFeature,
        values: Optional[List[str]] = None,
        marking_preference: Optional[str] = None,
    ) -> GrammarFeature:
        """
        Design a grammatical feature for the language.

        Args:
            feature_type: Type of feature to design (tense, aspect, etc.)
            values: Optional list of values for the feature
            marking_preference: How to mark the feature (prefix, suffix, etc.)

        Returns:
            Complete feature definition
        """
        system_prompt = """You are a constructed language (conlang) specialist helping design grammar.
Design a grammatical feature that:
1. Fits the phonological patterns of the existing vocabulary
2. Uses a consistent marking strategy
3. Has clear, memorable markers

Consider typological tendencies and make the feature naturalistic but learnable."""

        grammar_summary = self._get_existing_grammar_summary()

        user_content = f"""Design a {feature_type.value} feature:
{f'Values: {values}' if values else 'Suggest appropriate values'}
{f'Marking preference: {marking_preference}' if marking_preference else ''}

Existing grammar for reference:
{grammar_summary}"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=GrammarFeature
        )

        return response.content

    def generate_sentence_type_code(
        self,
        template: SentenceTypeTemplate,
        include_examples: bool = True,
    ) -> str:
        """
        Generate Python code for a sentence type.

        Args:
            template: The sentence type template
            include_examples: Whether to include get_examples() method

        Returns:
            Python code string
        """
        _ = include_examples  # Used in prompt

        system_prompt = """You are a Python code generator specializing in Pydantic models.
Generate a complete Sentence subclass that:
1. Inherits from Sentence
2. Has properly typed fields using Pydantic Field
3. Implements __str__() to render the sentence
4. Implements get_examples() with realistic examples

Follow the patterns used in existing sentence types.
Use proper type hints and docstrings."""

        grammar_summary = self._get_existing_grammar_summary()

        user_content = f"""Generate Python code for this sentence type:

Name: {template.name}
Description: {template.description}
Word Order: {template.word_order.value}
Required Components: {template.required_components}
Optional Components: {template.optional_components}
Pydantic Fields: {template.pydantic_fields}
Rendering Rules: {template.rendering_rules}

Example English sentences:
{template.example_english}

Existing code for reference:
{grammar_summary}

Generate complete, runnable Python code."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=str
        )

        return response.content

    def complete_grammar_design(
        self,
        pattern_description: str,
        example_sentences: Optional[List[str]] = None,
    ) -> GrammarDesign:
        """
        Complete workflow: design a sentence type and generate code.

        Args:
            pattern_description: Description of the pattern to implement
            example_sentences: Optional example sentences

        Returns:
            Complete grammar design with template and code
        """
        system_prompt = """You are a constructed language (conlang) specialist and Python developer.
Design a complete sentence type implementation including:
1. A detailed template with all components and features
2. Working Python code that follows existing patterns
3. Test cases to verify the implementation

The code should be production-ready and follow best practices."""

        grammar_summary = self._get_existing_grammar_summary()

        user_content = f"""Create a complete sentence type for:
{pattern_description}

{f'Example sentences: {example_sentences}' if example_sentences else ''}

Existing grammar and code for reference:
{grammar_summary}"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=GrammarDesign
        )

        return response.content
