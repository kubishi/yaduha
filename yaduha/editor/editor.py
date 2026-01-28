"""
Language Editor

Main orchestrator for LLM-assisted language creation and editing.
Combines vocabulary, grammar, examples, and validation into a unified interface.
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

from yaduha.agent import Agent
from yaduha.language import (
    LoadedLanguage,
    load_language_from_path,
    load_language_from_git,
    LanguageLoadError,
)
from yaduha.editor.vocabulary import (
    VocabularyAssistant,
    VocabularySuggestion,
    WordCategory,
    MorphologyPattern,
)
from yaduha.editor.grammar import (
    GrammarHelper,
    SentenceTypeTemplate,
    GrammarFeature,
    GrammaticalFeature,
    WordOrder,
)
from yaduha.editor.examples import (
    ExampleGenerator,
    GeneratedExample,
    ExampleQuality,
)
from yaduha.editor.validation import (
    ValidationFeedback,
    ValidationReport,
)


class LanguageProject(BaseModel):
    """Represents a language project being edited."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path = Field(..., description="Path to the language directory")
    name: str = Field(..., description="Language name")
    code: str = Field(..., description="Language code")
    is_loaded: bool = Field(default=False, description="Whether the language is currently loaded")
    last_error: Optional[str] = Field(None, description="Last error encountered")


class EditSession(BaseModel):
    """Tracks changes made during an editing session."""
    vocabulary_added: List[VocabularySuggestion] = Field(default_factory=list)
    grammar_added: List[SentenceTypeTemplate] = Field(default_factory=list)
    examples_generated: List[GeneratedExample] = Field(default_factory=list)
    files_modified: List[str] = Field(default_factory=list)
    validation_issues: List[str] = Field(default_factory=list)


class LanguageEditor:
    """
    Main interface for LLM-assisted language creation and editing.

    Orchestrates the vocabulary assistant, grammar helper, example generator,
    and validation feedback to provide a unified editing experience.

    Usage:
        ```python
        from yaduha.agent.anthropic import AnthropicAgent
        from yaduha.editor import LanguageEditor

        agent = AnthropicAgent(model="claude-3-5-sonnet-20241022")
        editor = LanguageEditor(agent)

        # Load an existing language
        editor.load_language("/path/to/my-lang")

        # Or start a new language
        editor.create_language("/path/to/new-lang", "My Language", "myl")

        # Add vocabulary
        suggestion = editor.suggest_vocabulary("water", WordCategory.noun)
        editor.apply_vocabulary([suggestion])

        # Add a sentence type
        template = editor.design_sentence_type(
            "Questions with yes/no answers",
            ["Is the water cold?", "Did you see it?"]
        )
        code = editor.generate_sentence_type_code(template)

        # Generate examples
        examples = editor.generate_examples(count=10)

        # Validate and get feedback
        report = editor.validate()
        ```
    """

    def __init__(self, agent: Agent[Any]):
        """
        Initialize the language editor.

        Args:
            agent: The LLM agent to use for all editing operations
        """
        self.agent = agent
        self._language: Optional[LoadedLanguage] = None
        self._project: Optional[LanguageProject] = None
        self._session: EditSession = EditSession()

        # Initialize sub-components (lazy loading with language context)
        self._vocabulary: Optional[VocabularyAssistant] = None
        self._grammar: Optional[GrammarHelper] = None
        self._examples: Optional[ExampleGenerator] = None
        self._validation: Optional[ValidationFeedback] = None

    @property
    def language(self) -> Optional[LoadedLanguage]:
        """Get the currently loaded language."""
        return self._language

    @property
    def project(self) -> Optional[LanguageProject]:
        """Get the current project."""
        return self._project

    @property
    def session(self) -> EditSession:
        """Get the current editing session."""
        return self._session

    @property
    def vocabulary(self) -> VocabularyAssistant:
        """Get the vocabulary assistant."""
        if self._vocabulary is None:
            self._vocabulary = VocabularyAssistant(self.agent, self._language)
        return self._vocabulary

    @property
    def grammar(self) -> GrammarHelper:
        """Get the grammar helper."""
        if self._grammar is None:
            self._grammar = GrammarHelper(self.agent, self._language)
        return self._grammar

    @property
    def examples(self) -> ExampleGenerator:
        """Get the example generator."""
        if self._examples is None:
            self._examples = ExampleGenerator(self.agent, self._language)
        return self._examples

    @property
    def validation(self) -> ValidationFeedback:
        """Get the validation feedback system."""
        if self._validation is None:
            self._validation = ValidationFeedback(self.agent, self._language)
        return self._validation

    def _update_components(self) -> None:
        """Update all components with the current language."""
        if self._vocabulary:
            self._vocabulary.language = self._language
        if self._grammar:
            self._grammar.language = self._language
        if self._examples:
            self._examples.language = self._language
        if self._validation:
            self._validation.language = self._language

    def load_language(self, path: Union[str, Path]) -> LoadedLanguage:
        """
        Load an existing language for editing.

        Args:
            path: Path to the language directory

        Returns:
            The loaded language

        Raises:
            LanguageLoadError: If loading fails
        """
        path = Path(path).resolve()

        try:
            self._language = load_language_from_path(path)
            self._project = LanguageProject(
                path=path,
                name=self._language.name,
                code=self._language.code,
                is_loaded=True,
                last_error=None,
            )
            self._update_components()
            self._session = EditSession()
            return self._language

        except LanguageLoadError as e:
            self._project = LanguageProject(
                path=path,
                name=path.name,
                code="unknown",
                is_loaded=False,
                last_error=str(e),
            )
            raise

    def load_language_from_git(self, repo_url: str) -> LoadedLanguage:
        """
        Load a language from a Git repository.

        Args:
            repo_url: URL of the Git repository

        Returns:
            The loaded language
        """
        lang, local_path = load_language_from_git(repo_url)
        self._language = lang
        self._project = LanguageProject(
            path=local_path,
            name=lang.name,
            code=lang.code,
            is_loaded=True,
            last_error=None,
        )
        self._update_components()
        self._session = EditSession()
        return lang

    def create_language(
        self,
        path: Union[str, Path],
        name: str,
        code: str,
        description: str = "",
        word_order: WordOrder = WordOrder.svo,
    ) -> Path:
        """
        Create a new language project with starter files.

        Args:
            path: Directory to create the language in
            name: Name of the language
            code: Short code for the language
            description: Optional description
            word_order: Default word order for the language

        Returns:
            Path to the created language directory
        """
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)

        # Create __init__.py with starter content
        init_content = f'''"""
{name} Language Definition

{description}
"""

from typing import List, Tuple, Type
from pydantic import BaseModel, Field
from yaduha.language import Sentence, VocabEntry

# Import vocabulary
from {path.name}.vocab import NOUNS, TRANSITIVE_VERBS, INTRANSITIVE_VERBS

# Language metadata
LANGUAGE_NAME = "{name}"
LANGUAGE_CODE = "{code}"
LANGUAGE_DESCRIPTION = """{description}"""


# ============================================================================
# SENTENCE TYPES
# Define your sentence structures here
# ============================================================================

class SimpleSentence(Sentence["SimpleSentence"]):
    """
    A basic {word_order.value} sentence.

    Example: "The cat sleeps."
    """
    subject: str = Field(..., description="The subject of the sentence")
    verb: str = Field(..., description="The verb")

    def __str__(self) -> str:
        # TODO: Implement rendering to {name}
        return f"{{self.subject}} {{self.verb}}"

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "SimpleSentence"]]:
        return [
            ("The cat sleeps.", SimpleSentence(subject="cat", verb="sleep")),
        ]


# Export sentence types
SENTENCE_TYPES = (SimpleSentence,)
'''

        init_file = path / "__init__.py"
        init_file.write_text(init_content)

        # Create vocab.py with starter content
        vocab_content = '''"""
Vocabulary for the language.

Add your vocabulary entries here.
"""

from yaduha.language import VocabEntry

# Nouns
NOUNS = [
    VocabEntry(english="cat", target="TODO"),
    VocabEntry(english="dog", target="TODO"),
    VocabEntry(english="water", target="TODO"),
]

# Transitive Verbs (verbs that take an object)
TRANSITIVE_VERBS = [
    VocabEntry(english="see", target="TODO"),
    VocabEntry(english="eat", target="TODO"),
]

# Intransitive Verbs (verbs without an object)
INTRANSITIVE_VERBS = [
    VocabEntry(english="sleep", target="TODO"),
    VocabEntry(english="run", target="TODO"),
]

# Adjectives
ADJECTIVES = [
    VocabEntry(english="big", target="TODO"),
    VocabEntry(english="small", target="TODO"),
]

# Adverbs
ADVERBS = [
    VocabEntry(english="quickly", target="TODO"),
    VocabEntry(english="slowly", target="TODO"),
]
'''

        vocab_file = path / "vocab.py"
        vocab_file.write_text(vocab_content)

        self._project = LanguageProject(
            path=path,
            name=name,
            code=code,
            is_loaded=False,
            last_error=None,
        )

        self._session = EditSession()
        self._session.files_modified.extend([str(init_file), str(vocab_file)])

        return path

    # =========================================================================
    # Vocabulary Operations
    # =========================================================================

    def suggest_vocabulary(
        self,
        english_word: str,
        category: WordCategory,
        context: Optional[str] = None,
    ) -> VocabularySuggestion:
        """
        Get a vocabulary suggestion for a word.

        Args:
            english_word: English word to translate
            category: Grammatical category
            context: Optional usage context

        Returns:
            Vocabulary suggestion with rationale
        """
        return self.vocabulary.suggest_word(english_word, category, context)

    def suggest_vocabulary_batch(
        self,
        words: List[tuple[str, WordCategory]],
    ) -> List[VocabularySuggestion]:
        """
        Get vocabulary suggestions for multiple words.

        Args:
            words: List of (english_word, category) tuples

        Returns:
            List of suggestions
        """
        result = self.vocabulary.suggest_batch(words)
        return result.suggestions

    def analyze_vocabulary_patterns(self) -> List[MorphologyPattern]:
        """
        Analyze existing vocabulary for morphological patterns.

        Returns:
            List of detected patterns
        """
        return self.vocabulary.analyze_patterns()

    def apply_vocabulary(
        self,
        suggestions: List[VocabularySuggestion],
        vocab_file: Optional[Path] = None,
    ) -> str:
        """
        Generate code for vocabulary additions.

        Args:
            suggestions: Vocabulary suggestions to apply
            vocab_file: Optional specific vocab file to update

        Returns:
            Generated Python code
        """
        code = self.vocabulary.generate_vocab_code(suggestions)
        self._session.vocabulary_added.extend(suggestions)
        return code

    # =========================================================================
    # Grammar Operations
    # =========================================================================

    def analyze_grammar(self):
        """
        Analyze existing grammar coverage.

        Returns:
            Analysis with gaps and suggestions
        """
        return self.grammar.analyze_grammar()

    def design_sentence_type(
        self,
        description: str,
        example_sentences: Optional[List[str]] = None,
        word_order: Optional[WordOrder] = None,
    ) -> SentenceTypeTemplate:
        """
        Design a new sentence type.

        Args:
            description: What the sentence type should express
            example_sentences: Optional example English sentences
            word_order: Optional word order

        Returns:
            Template for the sentence type
        """
        template = self.grammar.suggest_sentence_type(
            description, example_sentences, word_order
        )
        self._session.grammar_added.append(template)
        return template

    def design_grammar_feature(
        self,
        feature_type: GrammaticalFeature,
        values: Optional[List[str]] = None,
    ) -> GrammarFeature:
        """
        Design a grammatical feature (tense, aspect, etc.).

        Args:
            feature_type: Type of feature to design
            values: Optional values for the feature

        Returns:
            Feature definition
        """
        return self.grammar.design_grammar_feature(feature_type, values)

    def generate_sentence_type_code(
        self,
        template: SentenceTypeTemplate,
    ) -> str:
        """
        Generate Python code for a sentence type.

        Args:
            template: The sentence type template

        Returns:
            Python code
        """
        return self.grammar.generate_sentence_type_code(template)

    # =========================================================================
    # Example Operations
    # =========================================================================

    def generate_examples(
        self,
        sentence_type: Optional[str] = None,
        count: int = 5,
        quality: ExampleQuality = ExampleQuality.moderate,
    ) -> List[GeneratedExample]:
        """
        Generate example sentences.

        Args:
            sentence_type: Optional specific sentence type
            count: Number of examples
            quality: Complexity level

        Returns:
            List of generated examples
        """
        result = self.examples.generate_examples(sentence_type, count, quality)
        self._session.examples_generated.extend(result.examples)
        return result.examples

    def analyze_example_coverage(self):
        """
        Analyze diversity of existing examples.

        Returns:
            Coverage analysis with gaps
        """
        return self.examples.analyze_example_diversity()

    def generate_fewshot_prompt(
        self,
        num_examples: int = 3,
        task_description: str = "Translate the following English sentence",
    ) -> str:
        """
        Generate a few-shot prompt for translation.

        Args:
            num_examples: Number of examples to include
            task_description: Task description

        Returns:
            Formatted prompt string
        """
        return self.examples.generate_fewshot_prompt(
            num_examples=num_examples,
            task_description=task_description,
        )

    # =========================================================================
    # Validation Operations
    # =========================================================================

    def validate(self) -> ValidationReport:
        """
        Validate the current language and get feedback.

        Returns:
            Validation report with issues and fixes
        """
        if not self._project:
            raise ValueError("No language project loaded")

        try:
            # Try to reload the language
            self._language = load_language_from_path(self._project.path)
            self._project.is_loaded = True
            self._project.last_error = None
            self._update_components()

            # Review the code for potential issues
            init_file = self._project.path / "__init__.py"
            if init_file.exists():
                code = init_file.read_text()
                return self.validation.review_code(code)

            return ValidationReport(
                is_valid=True,
                summary="Language loads successfully",
            )

        except LanguageLoadError as e:
            self._project.is_loaded = False
            self._project.last_error = str(e)
            return self.validation.explain_load_error(e, self._project.path)

    def explain_error(self, error: Exception) -> ValidationReport:
        """
        Get an explanation for any error.

        Args:
            error: The error to explain

        Returns:
            Validation report with explanation
        """
        if isinstance(error, LanguageLoadError):
            return self.validation.explain_load_error(
                error,
                self._project.path if self._project else None
            )

        return self.validation.explain_validation_errors(
            [str(error)],
            file_path=str(self._project.path) if self._project else None,
        )

    def get_checklist(self) -> List[str]:
        """
        Get a validation checklist for the language.

        Returns:
            List of items to check
        """
        return self.validation.generate_validation_checklist()

    # =========================================================================
    # Session Management
    # =========================================================================

    def reset_session(self) -> None:
        """Reset the editing session, clearing tracked changes."""
        self._session = EditSession()

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of changes made in this session.

        Returns:
            Dictionary with session statistics
        """
        return {
            "vocabulary_added": len(self._session.vocabulary_added),
            "grammar_added": len(self._session.grammar_added),
            "examples_generated": len(self._session.examples_generated),
            "files_modified": self._session.files_modified,
            "validation_issues": len(self._session.validation_issues),
        }
