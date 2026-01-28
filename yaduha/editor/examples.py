"""
Example Generator

Auto-generates example sentences for few-shot prompting, ensuring
diverse coverage of grammatical features and vocabulary.
"""

from typing import List, Optional, Dict, Any, Type, Tuple, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

from yaduha.agent import Agent
from yaduha.language import Sentence, LoadedLanguage

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class ExampleQuality(str, Enum):
    """Quality/complexity level of generated examples."""
    simple = "simple"  # Basic sentences with common vocabulary
    moderate = "moderate"  # More varied vocabulary and features
    complex = "complex"  # Full feature coverage, edge cases


class GeneratedExample(BaseModel):
    """A generated example sentence."""
    english: str = Field(..., description="English version of the sentence")
    target: str = Field(..., description="Target language version")
    sentence_type: str = Field(..., description="Name of the sentence type used")
    features_demonstrated: List[str] = Field(
        default_factory=list,
        description="Grammatical features shown in this example"
    )
    vocabulary_used: List[str] = Field(
        default_factory=list,
        description="Key vocabulary items used"
    )
    notes: Optional[str] = Field(default=None, description="Notes about this example")


class ExampleSet(BaseModel):
    """A set of generated examples for a specific purpose."""
    purpose: str = Field(..., description="What these examples are designed to demonstrate")
    examples: List[GeneratedExample] = Field(..., description="The generated examples")
    coverage_report: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Report of what features/vocabulary each example covers"
    )


class ExampleDiversityAnalysis(BaseModel):
    """Analysis of example diversity and coverage."""
    total_examples: int = Field(..., description="Total number of examples")
    sentence_type_coverage: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of examples per sentence type"
    )
    feature_coverage: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of examples demonstrating each feature"
    )
    vocabulary_coverage: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of uses per vocabulary category"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps in coverage"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for additional examples"
    )


class _ExampleList(BaseModel):
    """Internal model for example list response."""
    examples: List[GeneratedExample]


class _TestCase(BaseModel):
    """Internal model for test case response."""
    english: str
    structured_input: Dict[str, Any]
    expected_output: str


class _TestCaseList(BaseModel):
    """Internal model for test case list response."""
    test_cases: List[_TestCase]


class ExampleGenerator:
    """
    LLM-powered generator for example sentences.

    Creates diverse, high-quality examples for:
    - Few-shot prompting in translation
    - Testing sentence type implementations
    - Documentation and learning materials
    """

    def __init__(self, agent: Agent[Any], language: Optional[LoadedLanguage] = None):
        """
        Initialize the example generator.

        Args:
            agent: The LLM agent to use for generation
            language: Optional loaded language to generate examples for
        """
        self.agent = agent
        self.language = language

    def _get_language_context(self) -> str:
        """Get context about the language for example generation."""
        if not self.language:
            return "No language context available."

        lines = [f"# {self.language.name} ({self.language.code})"]

        if self.language.description:
            lines.append(f"\n{self.language.description}")

        # Sentence types
        lines.append("\n## Available Sentence Types")
        for st in self.language.sentence_types:
            lines.append(f"\n### {st.__name__}")
            if st.__doc__:
                lines.append(st.__doc__.strip())

            # Show existing examples
            try:
                examples = st.get_examples()
                lines.append("\nExisting examples:")
                for eng, sentence in examples:
                    lines.append(f"- \"{eng}\" → \"{str(sentence)}\"")
            except Exception:
                pass

        # Vocabulary summary
        lines.append("\n## Vocabulary")
        if self.language.nouns:
            lines.append(f"- {len(self.language.nouns)} nouns")
        if self.language.transitive_verbs:
            lines.append(f"- {len(self.language.transitive_verbs)} transitive verbs")
        if self.language.intransitive_verbs:
            lines.append(f"- {len(self.language.intransitive_verbs)} intransitive verbs")
        if self.language.adjectives:
            lines.append(f"- {len(self.language.adjectives)} adjectives")
        if self.language.adverbs:
            lines.append(f"- {len(self.language.adverbs)} adverbs")

        return "\n".join(lines)

    def _get_vocabulary_list(self) -> str:
        """Get a formatted list of available vocabulary."""
        if not self.language:
            return ""

        lines = []

        if self.language.nouns:
            lines.append("Nouns: " + ", ".join(e.english for e in self.language.nouns[:30]))
        if self.language.transitive_verbs:
            lines.append("Transitive verbs: " + ", ".join(e.english for e in self.language.transitive_verbs[:30]))
        if self.language.intransitive_verbs:
            lines.append("Intransitive verbs: " + ", ".join(e.english for e in self.language.intransitive_verbs[:30]))
        if self.language.adjectives:
            lines.append("Adjectives: " + ", ".join(e.english for e in self.language.adjectives[:30]))
        if self.language.adverbs:
            lines.append("Adverbs: " + ", ".join(e.english for e in self.language.adverbs[:30]))

        return "\n".join(lines)

    def generate_examples(
        self,
        sentence_type: Optional[str] = None,
        count: int = 5,
        quality: ExampleQuality = ExampleQuality.moderate,
        focus_features: Optional[List[str]] = None,
        avoid_vocabulary: Optional[List[str]] = None,
    ) -> ExampleSet:
        """
        Generate example sentences.

        Args:
            sentence_type: Specific sentence type to generate for (or None for all)
            count: Number of examples to generate
            quality: Complexity level of examples
            focus_features: Grammatical features to focus on
            avoid_vocabulary: Vocabulary to avoid using

        Returns:
            Set of generated examples
        """
        system_prompt = """You are a linguistic example generator for constructed languages.
Generate diverse, natural-sounding example sentences that:
1. Use only vocabulary available in the language
2. Demonstrate the specified grammatical features
3. Are varied in structure and meaning
4. Would be useful for few-shot prompting

Each example should clearly demonstrate specific features and use different vocabulary."""

        context = self._get_language_context()
        vocab_list = self._get_vocabulary_list()

        user_content = f"""Generate {count} example sentences.

Quality level: {quality.value}
{f'Sentence type: {sentence_type}' if sentence_type else 'Any sentence type'}
{f'Focus on features: {focus_features}' if focus_features else ''}
{f'Avoid vocabulary: {avoid_vocabulary}' if avoid_vocabulary else ''}

Language context:
{context}

Available vocabulary:
{vocab_list}

Generate diverse examples that cover different combinations of features."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=ExampleSet
        )

        return response.content

    def generate_for_sentence_type(
        self,
        sentence_type: Type[Sentence[Any]],
        count: int = 5,
        feature_combinations: Optional[List[Dict[str, Any]]] = None,
    ) -> List[GeneratedExample]:
        """
        Generate examples specifically for a sentence type class.

        Args:
            sentence_type: The Sentence subclass to generate for
            count: Number of examples to generate
            feature_combinations: Specific feature combinations to demonstrate

        Returns:
            List of generated examples
        """
        # Get existing examples for context
        existing_examples = []
        try:
            for eng, sentence in sentence_type.get_examples():
                existing_examples.append(f'("{eng}", "{str(sentence)}")')
        except Exception:
            pass

        # Get model fields
        fields_info = []
        if hasattr(sentence_type, 'model_fields'):
            for name, field in sentence_type.model_fields.items():
                fields_info.append(f"- {name}: {field.annotation}")

        system_prompt = """You are a linguistic example generator for constructed languages.
Generate examples that:
1. Match the exact structure of the sentence type
2. Use only valid field values
3. Cover diverse feature combinations
4. Would be useful for training/few-shot prompting"""

        context = self._get_language_context()
        vocab_list = self._get_vocabulary_list()

        user_content = f"""Generate {count} examples for {sentence_type.__name__}.

{sentence_type.__doc__ or ''}

Fields:
{chr(10).join(fields_info)}

Existing examples:
{chr(10).join(existing_examples)}

{f'Feature combinations to cover: {feature_combinations}' if feature_combinations else ''}

Available vocabulary:
{vocab_list}

Generate new, diverse examples different from the existing ones."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=_ExampleList
        )

        return response.content.examples

    def analyze_example_diversity(
        self,
        examples: Optional[List[Tuple[str, Sentence[Any]]]] = None,
    ) -> ExampleDiversityAnalysis:
        """
        Analyze the diversity and coverage of existing examples.

        Args:
            examples: Examples to analyze (or use language's examples if None)

        Returns:
            Analysis of example diversity
        """
        if examples is None and self.language:
            examples = self.language.get_all_examples()

        if not examples:
            return ExampleDiversityAnalysis(
                total_examples=0,
                gaps=["No examples to analyze"],
                suggestions=["Add initial examples using generate_examples()"]
            )

        system_prompt = """You are a linguistic analyst evaluating example sentence coverage.
Analyze the given examples and identify:
1. Which sentence types are represented
2. Which grammatical features are demonstrated
3. Which vocabulary categories are used
4. Gaps in coverage
5. Suggestions for improving diversity"""

        # Format examples for analysis
        example_text = []
        for eng, sentence in examples:
            example_text.append(f'Type: {type(sentence).__name__}')
            example_text.append(f'English: "{eng}"')
            example_text.append(f'Target: "{str(sentence)}"')
            example_text.append("")

        context = self._get_language_context()

        user_content = f"""Analyze these examples:

{chr(10).join(example_text)}

Language context:
{context}"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=ExampleDiversityAnalysis
        )

        return response.content

    def generate_fewshot_prompt(
        self,
        sentence_type: Optional[str] = None,
        num_examples: int = 3,
        task_description: str = "Translate the following English sentence",
    ) -> str:
        """
        Generate a complete few-shot prompt for translation.

        Args:
            sentence_type: Specific sentence type (or None for mixed)
            num_examples: Number of examples to include
            task_description: Description of the translation task

        Returns:
            Formatted few-shot prompt string
        """
        # Get examples
        examples = self.generate_examples(
            sentence_type=sentence_type,
            count=num_examples,
            quality=ExampleQuality.moderate
        )

        # Format as few-shot prompt
        lines = [
            f"# {self.language.name if self.language else 'Target Language'} Translation",
            "",
            task_description,
            "",
            "## Examples",
            ""
        ]

        for ex in examples.examples:
            lines.append(f"English: {ex.english}")
            lines.append(f"Translation: {ex.target}")
            lines.append("")

        lines.append("## Your Turn")
        lines.append("")
        lines.append("English: {input}")
        lines.append("Translation: ")

        return "\n".join(lines)

    def generate_test_cases(
        self,
        sentence_type: Type[Sentence[Any]],
        count: int = 10,
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Generate test cases for a sentence type.

        Args:
            sentence_type: The Sentence subclass to test
            count: Number of test cases

        Returns:
            List of (english, structured_input, expected_output) tuples
        """
        system_prompt = """You are a test case generator for constructed language implementations.
Generate test cases that:
1. Cover edge cases and boundary conditions
2. Use diverse vocabulary and feature combinations
3. Include both simple and complex sentences
4. Would catch common implementation bugs

Return structured input that matches the sentence type's Pydantic model."""

        # Get model fields
        fields_info = []
        if hasattr(sentence_type, 'model_fields'):
            for name, field in sentence_type.model_fields.items():
                fields_info.append(f"- {name}: {field.annotation}")

        context = self._get_language_context()
        vocab_list = self._get_vocabulary_list()

        user_content = f"""Generate {count} test cases for {sentence_type.__name__}.

{sentence_type.__doc__ or ''}

Fields:
{chr(10).join(fields_info)}

Available vocabulary:
{vocab_list}

For each test case, provide:
1. The English sentence
2. The structured input (as a dict matching the Pydantic model)
3. The expected output string"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=_TestCaseList
        )

        return [
            (tc.english, tc.structured_input, tc.expected_output)
            for tc in response.content.test_cases
        ]
