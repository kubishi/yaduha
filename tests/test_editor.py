"""
Tests for the LLM-assisted language editing tools.

These tests verify:
1. Editor module imports correctly
2. Data models are properly structured
3. Editor can be instantiated with mock agents
4. Basic non-LLM functionality works
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List, Tuple

from pydantic import BaseModel

from yaduha.language import Sentence, VocabEntry, LoadedLanguage
from yaduha.agent import Agent, AgentResponse


# ============================================================================
# Test Fixtures
# ============================================================================


class MockSentence(Sentence["MockSentence"]):
    """Mock sentence type for testing."""
    subject: str
    verb: str

    def __str__(self) -> str:
        return f"{self.subject} {self.verb}"

    @classmethod
    def get_examples(cls) -> List[Tuple[str, "MockSentence"]]:
        return [
            ("The cat sleeps.", MockSentence(subject="cat", verb="sleep")),
            ("A dog runs.", MockSentence(subject="dog", verb="run")),
        ]


@pytest.fixture
def mock_agent() -> Mock:
    """Create a mock agent for testing."""
    agent = Mock(spec=Agent)
    agent.model = "mock-model"

    # Default response for get_response
    def mock_get_response(messages, response_format=str, tools=None):
        if response_format == str:
            return AgentResponse(
                content="Mock response",
                response_time=0.1,
                prompt_tokens=10,
                completion_tokens=5,
            )
        # For Pydantic models, try to create a minimal instance
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Return a mock that has the expected structure
            mock_content = MagicMock(spec=response_format)
            return AgentResponse(
                content=mock_content,
                response_time=0.1,
                prompt_tokens=10,
                completion_tokens=5,
            )
        return AgentResponse(
            content="Mock response",
            response_time=0.1,
            prompt_tokens=10,
            completion_tokens=5,
        )

    agent.get_response = Mock(side_effect=mock_get_response)
    return agent


@pytest.fixture
def mock_language() -> LoadedLanguage:
    """Create a mock loaded language for testing."""
    mock_module = Mock()
    mock_module.LANGUAGE_NAME = "Test Language"
    mock_module.LANGUAGE_CODE = "test"

    return LoadedLanguage(
        name="Test Language",
        code="test",
        description="A test language",
        sentence_types=(MockSentence,),
        module=mock_module,
        nouns=[
            VocabEntry(english="cat", target="felis"),
            VocabEntry(english="dog", target="canis"),
            VocabEntry(english="water", target="aqua"),
        ],
        transitive_verbs=[
            VocabEntry(english="see", target="vid"),
            VocabEntry(english="eat", target="ed"),
        ],
        intransitive_verbs=[
            VocabEntry(english="sleep", target="dorm"),
            VocabEntry(english="run", target="curr"),
        ],
        adjectives=[
            VocabEntry(english="big", target="magn"),
            VocabEntry(english="small", target="parv"),
        ],
        adverbs=[
            VocabEntry(english="quickly", target="celer"),
        ],
    )


# ============================================================================
# Import Tests
# ============================================================================


class TestEditorImports:
    """Test that all editor modules import correctly."""

    def test_import_vocabulary(self):
        """Test vocabulary module imports."""
        from yaduha.editor.vocabulary import (
            VocabularyAssistant,
            VocabularySuggestion,
            WordCategory,
            MorphologyPattern,
        )
        assert VocabularyAssistant is not None
        assert VocabularySuggestion is not None
        assert WordCategory is not None
        assert MorphologyPattern is not None

    def test_import_grammar(self):
        """Test grammar module imports."""
        from yaduha.editor.grammar import (
            GrammarHelper,
            SentenceTypeTemplate,
            GrammarFeature,
            GrammaticalFeature,
            WordOrder,
        )
        assert GrammarHelper is not None
        assert SentenceTypeTemplate is not None
        assert GrammarFeature is not None
        assert GrammaticalFeature is not None
        assert WordOrder is not None

    def test_import_examples(self):
        """Test examples module imports."""
        from yaduha.editor.examples import (
            ExampleGenerator,
            GeneratedExample,
            ExampleQuality,
            ExampleSet,
        )
        assert ExampleGenerator is not None
        assert GeneratedExample is not None
        assert ExampleQuality is not None
        assert ExampleSet is not None

    def test_import_validation(self):
        """Test validation module imports."""
        from yaduha.editor.validation import (
            ValidationFeedback,
            ValidationIssue,
            ValidationFix,
            ValidationReport,
            IssueSeverity,
            IssueCategory,
        )
        assert ValidationFeedback is not None
        assert ValidationIssue is not None
        assert ValidationFix is not None
        assert ValidationReport is not None
        assert IssueSeverity is not None
        assert IssueCategory is not None

    def test_import_editor(self):
        """Test main editor module imports."""
        from yaduha.editor import (
            LanguageEditor,
            VocabularyAssistant,
            GrammarHelper,
            ExampleGenerator,
            ValidationFeedback,
        )
        assert LanguageEditor is not None
        assert VocabularyAssistant is not None
        assert GrammarHelper is not None
        assert ExampleGenerator is not None
        assert ValidationFeedback is not None


# ============================================================================
# Data Model Tests
# ============================================================================


class TestDataModels:
    """Test that data models can be instantiated correctly."""

    def test_vocabulary_suggestion_model(self):
        """Test VocabularySuggestion model."""
        from yaduha.editor.vocabulary import VocabularySuggestion, WordCategory

        suggestion = VocabularySuggestion(
            english="water",
            target="aqua",
            category=WordCategory.noun,
            rationale="Latin root for water-related terms",
            alternatives=["hydra", "aque"],
            related_words=["rain", "river"],
        )

        assert suggestion.english == "water"
        assert suggestion.target == "aqua"
        assert suggestion.category == WordCategory.noun
        assert len(suggestion.alternatives) == 2

    def test_morphology_pattern_model(self):
        """Test MorphologyPattern model."""
        from yaduha.editor.vocabulary import MorphologyPattern

        pattern = MorphologyPattern(
            name="Vowel Harmony",
            description="Back vowels trigger back vowel suffixes",
            examples=["casa-rum", "domo-rum"],
        )

        assert pattern.name == "Vowel Harmony"
        assert len(pattern.examples) == 2

    def test_sentence_type_template_model(self):
        """Test SentenceTypeTemplate model."""
        from yaduha.editor.grammar import SentenceTypeTemplate, WordOrder

        template = SentenceTypeTemplate(
            name="QuestionSentence",
            description="Yes/no questions",
            word_order=WordOrder.vso,
            required_components=["verb", "subject"],
            optional_components=["object"],
            example_english=["Is the cat sleeping?"],
        )

        assert template.name == "QuestionSentence"
        assert template.word_order == WordOrder.vso
        assert "verb" in template.required_components

    def test_grammar_feature_model(self):
        """Test GrammarFeature model."""
        from yaduha.editor.grammar import GrammarFeature, GrammaticalFeature

        feature = GrammarFeature(
            name="Tense",
            feature_type=GrammaticalFeature.tense,
            description="Marks when an action occurs",
            values=["past", "present", "future"],
            marking_strategy="suffix",
            markers={"past": "-ed", "present": "-ing", "future": "-will"},
            applies_to=["verb"],
        )

        assert feature.name == "Tense"
        assert feature.feature_type == GrammaticalFeature.tense
        assert len(feature.values) == 3
        assert feature.markers["past"] == "-ed"

    def test_generated_example_model(self):
        """Test GeneratedExample model."""
        from yaduha.editor.examples import GeneratedExample

        example = GeneratedExample(
            english="The cat sleeps.",
            target="felis dormit",
            sentence_type="SubjectVerbSentence",
            features_demonstrated=["present tense", "third person"],
            vocabulary_used=["cat", "sleep"],
        )

        assert example.english == "The cat sleeps."
        assert example.sentence_type == "SubjectVerbSentence"
        assert "present tense" in example.features_demonstrated

    def test_validation_issue_model(self):
        """Test ValidationIssue model."""
        from yaduha.editor.validation import (
            ValidationIssue,
            IssueSeverity,
            IssueCategory,
        )

        issue = ValidationIssue(
            severity=IssueSeverity.error,
            category=IssueCategory.missing_export,
            location="__init__.py:1",
            message="Missing LANGUAGE_NAME",
            explanation="Your language needs a name defined",
        )

        assert issue.severity == IssueSeverity.error
        assert issue.category == IssueCategory.missing_export

    def test_validation_fix_model(self):
        """Test ValidationFix model."""
        from yaduha.editor.validation import ValidationFix

        fix = ValidationFix(
            issue_summary="Missing LANGUAGE_NAME",
            explanation="Add a name for your language",
            code_before="# empty",
            code_after='LANGUAGE_NAME = "My Language"',
            file_path="__init__.py",
            steps=["Open __init__.py", "Add the LANGUAGE_NAME line"],
        )

        assert fix.issue_summary == "Missing LANGUAGE_NAME"
        assert len(fix.steps) == 2


# ============================================================================
# Component Initialization Tests
# ============================================================================


class TestComponentInitialization:
    """Test that editor components can be initialized."""

    def test_vocabulary_assistant_init(self, mock_agent, mock_language):
        """Test VocabularyAssistant initialization."""
        from yaduha.editor.vocabulary import VocabularyAssistant

        assistant = VocabularyAssistant(mock_agent, mock_language)

        assert assistant.agent is mock_agent
        assert assistant.language is mock_language

    def test_vocabulary_assistant_no_language(self, mock_agent):
        """Test VocabularyAssistant works without a language."""
        from yaduha.editor.vocabulary import VocabularyAssistant

        assistant = VocabularyAssistant(mock_agent)

        assert assistant.agent is mock_agent
        assert assistant.language is None

    def test_grammar_helper_init(self, mock_agent, mock_language):
        """Test GrammarHelper initialization."""
        from yaduha.editor.grammar import GrammarHelper

        helper = GrammarHelper(mock_agent, mock_language)

        assert helper.agent is mock_agent
        assert helper.language is mock_language

    def test_example_generator_init(self, mock_agent, mock_language):
        """Test ExampleGenerator initialization."""
        from yaduha.editor.examples import ExampleGenerator

        generator = ExampleGenerator(mock_agent, mock_language)

        assert generator.agent is mock_agent
        assert generator.language is mock_language

    def test_validation_feedback_init(self, mock_agent, mock_language):
        """Test ValidationFeedback initialization."""
        from yaduha.editor.validation import ValidationFeedback

        feedback = ValidationFeedback(mock_agent, mock_language)

        assert feedback.agent is mock_agent
        assert feedback.language is mock_language


# ============================================================================
# LanguageEditor Tests
# ============================================================================


class TestLanguageEditor:
    """Test the main LanguageEditor class."""

    def test_editor_init(self, mock_agent):
        """Test LanguageEditor initialization."""
        from yaduha.editor import LanguageEditor

        editor = LanguageEditor(mock_agent)

        assert editor.agent is mock_agent
        assert editor.language is None
        assert editor.project is None

    def test_editor_lazy_loading(self, mock_agent):
        """Test that editor components are lazy loaded."""
        from yaduha.editor import LanguageEditor

        editor = LanguageEditor(mock_agent)

        # Components should not be created until accessed
        assert editor._vocabulary is None
        assert editor._grammar is None
        assert editor._examples is None
        assert editor._validation is None

        # Accessing triggers creation
        _ = editor.vocabulary
        assert editor._vocabulary is not None

        _ = editor.grammar
        assert editor._grammar is not None

    def test_editor_create_language(self, mock_agent, tmp_path):
        """Test creating a new language project."""
        from yaduha.editor import LanguageEditor

        editor = LanguageEditor(mock_agent)

        lang_path = tmp_path / "my_lang"
        result = editor.create_language(
            path=lang_path,
            name="My Language",
            code="myl",
            description="A test language",
        )

        assert result == lang_path
        assert lang_path.exists()
        assert (lang_path / "__init__.py").exists()
        assert (lang_path / "vocab.py").exists()

        # Check that project is set
        assert editor.project is not None
        assert editor.project.name == "My Language"
        assert editor.project.code == "myl"

    def test_editor_create_language_content(self, mock_agent, tmp_path):
        """Test that created language files have correct content."""
        from yaduha.editor import LanguageEditor

        editor = LanguageEditor(mock_agent)

        lang_path = tmp_path / "test_lang"
        editor.create_language(
            path=lang_path,
            name="Test Language",
            code="tst",
        )

        # Check __init__.py content
        init_content = (lang_path / "__init__.py").read_text()
        assert 'LANGUAGE_NAME = "Test Language"' in init_content
        assert 'LANGUAGE_CODE = "tst"' in init_content
        assert "SENTENCE_TYPES" in init_content

        # Check vocab.py content
        vocab_content = (lang_path / "vocab.py").read_text()
        assert "VocabEntry" in vocab_content
        assert "NOUNS" in vocab_content
        assert "TRANSITIVE_VERBS" in vocab_content

    def test_editor_session_tracking(self, mock_agent):
        """Test that editor tracks session changes."""
        from yaduha.editor import LanguageEditor

        editor = LanguageEditor(mock_agent)

        # Session should be initialized
        assert editor.session is not None
        assert len(editor.session.vocabulary_added) == 0
        assert len(editor.session.grammar_added) == 0

        # Get summary
        summary = editor.get_session_summary()
        assert summary["vocabulary_added"] == 0
        assert summary["grammar_added"] == 0

    def test_editor_reset_session(self, mock_agent, tmp_path):
        """Test resetting the editor session."""
        from yaduha.editor import LanguageEditor

        editor = LanguageEditor(mock_agent)

        # Create a language (which modifies session)
        editor.create_language(tmp_path / "lang", "Test", "tst")

        # Session should have modified files
        assert len(editor.session.files_modified) > 0

        # Reset
        editor.reset_session()

        # Session should be clean
        assert len(editor.session.files_modified) == 0


# ============================================================================
# Vocabulary Code Generation Tests
# ============================================================================


class TestVocabCodeGeneration:
    """Test vocabulary code generation."""

    def test_generate_vocab_code(self, mock_agent, mock_language):
        """Test generating vocab.py code from suggestions."""
        from yaduha.editor.vocabulary import (
            VocabularyAssistant,
            VocabularySuggestion,
            WordCategory,
        )

        assistant = VocabularyAssistant(mock_agent, mock_language)

        suggestions = [
            VocabularySuggestion(
                english="tree",
                target="arbor",
                category=WordCategory.noun,
                rationale="Latin root",
            ),
            VocabularySuggestion(
                english="grow",
                target="cresc",
                category=WordCategory.intransitive_verb,
                rationale="Latin root",
            ),
        ]

        code = assistant.generate_vocab_code(suggestions)

        assert "from yaduha.language import VocabEntry" in code
        assert 'VocabEntry(english="tree", target="arbor")' in code
        assert 'VocabEntry(english="grow", target="cresc")' in code
        assert "NOUNS" in code
        assert "INTRANSITIVE_VERBS" in code


# ============================================================================
# Enums Tests
# ============================================================================


class TestEnums:
    """Test enum values."""

    def test_word_category_values(self):
        """Test WordCategory enum."""
        from yaduha.editor.vocabulary import WordCategory

        assert WordCategory.noun.value == "noun"
        assert WordCategory.transitive_verb.value == "transitive_verb"
        assert WordCategory.intransitive_verb.value == "intransitive_verb"
        assert WordCategory.adjective.value == "adjective"
        assert WordCategory.adverb.value == "adverb"

    def test_word_order_values(self):
        """Test WordOrder enum."""
        from yaduha.editor.grammar import WordOrder

        assert WordOrder.svo.value == "SVO"
        assert WordOrder.sov.value == "SOV"
        assert WordOrder.vso.value == "VSO"
        assert WordOrder.vos.value == "VOS"
        assert WordOrder.free.value == "free"

    def test_grammatical_feature_values(self):
        """Test GrammaticalFeature enum."""
        from yaduha.editor.grammar import GrammaticalFeature

        assert GrammaticalFeature.tense.value == "tense"
        assert GrammaticalFeature.aspect.value == "aspect"
        assert GrammaticalFeature.mood.value == "mood"
        assert GrammaticalFeature.evidentiality.value == "evidentiality"
        assert GrammaticalFeature.case.value == "case"

    def test_example_quality_values(self):
        """Test ExampleQuality enum."""
        from yaduha.editor.examples import ExampleQuality

        assert ExampleQuality.simple.value == "simple"
        assert ExampleQuality.moderate.value == "moderate"
        assert ExampleQuality.complex.value == "complex"

    def test_issue_severity_values(self):
        """Test IssueSeverity enum."""
        from yaduha.editor.validation import IssueSeverity

        assert IssueSeverity.error.value == "error"
        assert IssueSeverity.warning.value == "warning"
        assert IssueSeverity.suggestion.value == "suggestion"

    def test_issue_category_values(self):
        """Test IssueCategory enum."""
        from yaduha.editor.validation import IssueCategory

        assert IssueCategory.missing_export.value == "missing_export"
        assert IssueCategory.invalid_type.value == "invalid_type"
        assert IssueCategory.example_error.value == "example_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
