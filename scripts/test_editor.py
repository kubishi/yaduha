#!/usr/bin/env python3
"""
Test script for the LLM-assisted language editor.

This script demonstrates the capabilities of the language editor tools:
- Creating a new language project
- Getting vocabulary suggestions
- Designing sentence types
- Generating examples
- Validating and getting feedback

Usage:
    python scripts/test_editor.py [--live] [--provider anthropic|openai|ollama]

Options:
    --live          Run live LLM tests (requires API key)
    --provider      Which LLM provider to use (default: anthropic)
"""

import argparse
import os
import sys
from pathlib import Path
import tempfile
import shutil
import dotenv

from yaduha.editor import (
    LanguageEditor,
    VocabularyAssistant,
    GrammarHelper,
    ExampleGenerator,
    ValidationFeedback,
)
from yaduha.editor.vocabulary import WordCategory, VocabularySuggestion
from yaduha.editor.grammar import WordOrder, GrammaticalFeature
from yaduha.editor.examples import ExampleQuality
from yaduha.editor.validation import IssueSeverity, IssueCategory
from yaduha.language import load_language_from_git


# Load environment variables from .env file if it exists
dotenv.load_dotenv()

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    # Test main editor
    from yaduha.editor import LanguageEditor
    print("  ✓ LanguageEditor")

    # Test vocabulary
    from yaduha.editor.vocabulary import (
        VocabularyAssistant,
        VocabularySuggestion,
        VocabularySuggestions,
        MorphologyPattern,
        WordCategory,
    )
    print("  ✓ Vocabulary components")

    # Test grammar
    from yaduha.editor.grammar import (
        GrammarHelper,
        SentenceTypeTemplate,
        GrammarFeature,
        GrammaticalFeature,
        WordOrder,
        SentenceTypeAnalysis,
        GrammarDesign,
    )
    print("  ✓ Grammar components")

    # Test examples
    from yaduha.editor.examples import (
        ExampleGenerator,
        GeneratedExample,
        ExampleSet,
        ExampleQuality,
        ExampleDiversityAnalysis,
    )
    print("  ✓ Example components")

    # Test validation
    from yaduha.editor.validation import (
        ValidationFeedback,
        ValidationIssue,
        ValidationFix,
        ValidationReport,
        IssueSeverity,
        IssueCategory,
    )
    print("  ✓ Validation components")

    print("\nAll imports successful!")
    return True


def test_data_models():
    """Test that data models can be instantiated."""
    print("\n" + "=" * 60)
    print("Testing data models...")
    print("=" * 60)

    # Test VocabularySuggestion
    suggestion = VocabularySuggestion(
        english="water",
        target="akwa",
        category=WordCategory.noun,
        rationale="Based on Latin 'aqua'",
        alternatives=["hydra", "wata"],
        related_words=["rain", "river", "ocean"],
    )
    print(f"  ✓ VocabularySuggestion: {suggestion.english} → {suggestion.target}")

    # Test SentenceTypeTemplate
    from yaduha.editor.grammar import SentenceTypeTemplate
    template = SentenceTypeTemplate(
        name="QuestionSentence",
        description="Yes/no questions",
        word_order=WordOrder.vso,
        required_components=["verb", "subject"],
        optional_components=["object"],
        example_english=["Is it raining?", "Do you see the cat?"],
    )
    print(f"  ✓ SentenceTypeTemplate: {template.name} ({template.word_order.value})")

    # Test GeneratedExample
    from yaduha.editor.examples import GeneratedExample
    example = GeneratedExample(
        english="The cat sleeps.",
        target="felis dormit",
        sentence_type="SubjectVerbSentence",
        features_demonstrated=["present tense", "third person"],
        vocabulary_used=["cat", "sleep"],
    )
    print(f"  ✓ GeneratedExample: '{example.english}' → '{example.target}'")

    # Test ValidationIssue
    from yaduha.editor.validation import ValidationIssue
    issue = ValidationIssue(
        severity=IssueSeverity.error,
        category=IssueCategory.missing_export,
        location="__init__.py:1",
        message="Missing LANGUAGE_NAME",
        explanation="Your language needs a name",
    )
    print(f"  ✓ ValidationIssue: [{issue.severity.value}] {issue.message}")

    print("\nAll data models work!")
    return True


def test_create_language(temp_dir: Path):
    """Test creating a new language project."""
    print("\n" + "=" * 60)
    print("Testing language creation...")
    print("=" * 60)

    from unittest.mock import Mock
    from yaduha.agent import Agent

    # Create mock agent
    mock_agent = Mock(spec=Agent)
    mock_agent.model = "mock-model"

    # Create editor
    editor = LanguageEditor(mock_agent)

    # Create a new language
    lang_path = temp_dir / "test_lang"
    result = editor.create_language(
        path=lang_path,
        name="Test Language",
        code="tst",
        description="A test constructed language",
        word_order=WordOrder.svo,
    )

    print(f"  Created language at: {result}")

    # Check files were created
    assert (lang_path / "__init__.py").exists(), "Missing __init__.py"
    print("  ✓ __init__.py created")

    assert (lang_path / "vocab.py").exists(), "Missing vocab.py"
    print("  ✓ vocab.py created")

    # Check content
    init_content = (lang_path / "__init__.py").read_text()
    assert 'LANGUAGE_NAME = "Test Language"' in init_content
    assert 'LANGUAGE_CODE = "tst"' in init_content
    assert "SENTENCE_TYPES" in init_content
    print("  ✓ __init__.py has correct content")

    vocab_content = (lang_path / "vocab.py").read_text()
    assert "NOUNS" in vocab_content
    assert "TRANSITIVE_VERBS" in vocab_content
    print("  ✓ vocab.py has correct content")

    # Check project state
    assert editor.project is not None
    assert editor.project.name == "Test Language"
    assert editor.project.code == "tst"
    print(f"  ✓ Project state: {editor.project.name} ({editor.project.code})")

    # Check session tracking
    assert len(editor.session.files_modified) == 2
    print(f"  ✓ Session tracked {len(editor.session.files_modified)} files")

    print("\nLanguage creation successful!")
    return True


def test_vocab_code_generation():
    """Test vocabulary code generation (no LLM needed)."""
    print("\n" + "=" * 60)
    print("Testing vocabulary code generation...")
    print("=" * 60)

    from unittest.mock import Mock
    from yaduha.agent import Agent

    mock_agent = Mock(spec=Agent)
    assistant = VocabularyAssistant(mock_agent)

    # Create some suggestions
    suggestions = [
        VocabularySuggestion(
            english="sun",
            target="sola",
            category=WordCategory.noun,
            rationale="Test",
        ),
        VocabularySuggestion(
            english="moon",
            target="luna",
            category=WordCategory.noun,
            rationale="Test",
        ),
        VocabularySuggestion(
            english="shine",
            target="brila",
            category=WordCategory.intransitive_verb,
            rationale="Test",
        ),
    ]

    # Generate code
    code = assistant.generate_vocab_code(suggestions)

    print("Generated code:")
    print("-" * 40)
    print(code)
    print("-" * 40)

    # Verify code
    assert "from yaduha.language import VocabEntry" in code
    assert 'english="sun"' in code
    assert 'target="sola"' in code
    assert "NOUNS" in code
    assert "INTRANSITIVE_VERBS" in code

    print("  ✓ Code generation successful!")
    return True


def test_editor_with_loaded_language():
    """Test editor with a real loaded language."""
    print("\n" + "=" * 60)
    print("Testing editor with loaded language...")
    print("=" * 60)

    from unittest.mock import Mock
    from yaduha.agent import Agent

    # Load OVP language
    print("  Loading OVP language from git...")
    try:
        lang, lang_path = load_language_from_git(
            "https://github.com/kubishi/ovp-lang",
            verbose=False,
        )
        print(f"  ✓ Loaded {lang.name} ({lang.code})")
        print(f"    - {len(lang.sentence_types)} sentence types")
        print(f"    - {len(lang.nouns)} nouns")
        print(f"    - {len(lang.transitive_verbs)} transitive verbs")
    except Exception as e:
        print(f"  ⚠ Could not load OVP language: {e}")
        print("  Skipping this test (requires network)")
        return True

    # Create editor with mock agent
    mock_agent = Mock(spec=Agent)
    mock_agent.model = "mock-model"

    # Initialize components with the loaded language
    vocab_assistant = VocabularyAssistant(mock_agent, lang)
    grammar_helper = GrammarHelper(mock_agent, lang)
    example_gen = ExampleGenerator(mock_agent, lang)
    validation = ValidationFeedback(mock_agent, lang)

    # Test that context methods work
    vocab_summary = vocab_assistant._get_existing_vocab_summary()
    assert "Existing Vocabulary" in vocab_summary
    assert lang.name in vocab_summary
    print("  ✓ Vocabulary summary generated")

    grammar_summary = grammar_helper._get_existing_grammar_summary()
    assert "Existing Grammar" in grammar_summary
    assert "Sentence Types" in grammar_summary
    print("  ✓ Grammar summary generated")

    lang_context = example_gen._get_language_context()
    assert lang.name in lang_context
    print("  ✓ Language context generated")

    validation_context = validation._get_language_context()
    assert lang.name in validation_context
    print("  ✓ Validation context generated")

    print("\nEditor with loaded language works!")
    return True


def get_agent(provider: str):
    """Get an LLM agent based on provider."""
    if provider == "anthropic":
        from yaduha.agent.anthropic import AnthropicAgent
        return AnthropicAgent(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-3-5-sonnet-20241022"
        )
    elif provider == "openai":
        from yaduha.agent.openai import OpenAIAgent
        return OpenAIAgent(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini"
        )
    elif provider == "ollama":
        from yaduha.agent.ollama import OllamaAgent
        return OllamaAgent(model="llama3.2")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def test_live_vocabulary(agent):
    """Test live vocabulary suggestions."""
    print("\n" + "=" * 60)
    print("Testing LIVE vocabulary suggestions...")
    print("=" * 60)

    # Load a language for context
    try:
        lang, _ = load_language_from_git(
            "https://github.com/kubishi/ovp-lang",
            verbose=False,
        )
    except Exception as e:
        print(f"  ⚠ Could not load language: {e}")
        return False

    assistant = VocabularyAssistant(agent, lang)

    # Test pattern analysis
    print("\n  Analyzing vocabulary patterns...")
    try:
        patterns = assistant.analyze_patterns()
        print(f"  ✓ Found {len(patterns)} patterns:")
        for p in patterns[:3]:
            print(f"    - {p.name}: {p.description[:60]}...")
    except Exception as e:
        print(f"  ⚠ Pattern analysis failed: {e}")

    # Test word suggestion
    print("\n  Suggesting vocabulary...")
    try:
        suggestion = assistant.suggest_word(
            english_word="mountain",
            category=WordCategory.noun,
            context="A tall natural elevation of earth",
        )
        print(f"  ✓ Suggestion: '{suggestion.english}' → '{suggestion.target}'")
        print(f"    Rationale: {suggestion.rationale[:80]}...")
        if suggestion.alternatives:
            print(f"    Alternatives: {', '.join(suggestion.alternatives[:3])}")
    except Exception as e:
        print(f"  ⚠ Word suggestion failed: {e}")

    return True


def test_live_grammar(agent):
    """Test live grammar design."""
    print("\n" + "=" * 60)
    print("Testing LIVE grammar design...")
    print("=" * 60)

    # Load a language for context
    try:
        lang, _ = load_language_from_git(
            "https://github.com/kubishi/ovp-lang",
            verbose=False,
        )
    except Exception as e:
        print(f"  ⚠ Could not load language: {e}")
        return False

    helper = GrammarHelper(agent, lang)

    # Test grammar analysis
    print("\n  Analyzing existing grammar...")
    try:
        analysis = helper.analyze_grammar()
        print(f"  ✓ Analysis complete:")
        print(f"    Existing types: {', '.join(analysis.existing_types[:3])}")
        print(f"    Gaps: {', '.join(analysis.gaps[:3])}")
        print(f"    Suggestions: {', '.join(analysis.suggestions[:2])}")
    except Exception as e:
        print(f"  ⚠ Grammar analysis failed: {e}")

    # Test sentence type design
    print("\n  Designing a question sentence type...")
    try:
        template = helper.suggest_sentence_type(
            pattern_description="Yes/no questions that can be answered with 'yes' or 'no'",
            example_sentences=["Is the dog sleeping?", "Did you see it?"],
        )
        print(f"  ✓ Template: {template.name}")
        print(f"    Word order: {template.word_order.value}")
        print(f"    Required: {', '.join(template.required_components)}")
        print(f"    Description: {template.description[:60]}...")
    except Exception as e:
        print(f"  ⚠ Sentence type design failed: {e}")

    return True


def test_live_examples(agent):
    """Test live example generation."""
    print("\n" + "=" * 60)
    print("Testing LIVE example generation...")
    print("=" * 60)

    # Load a language for context
    try:
        lang, _ = load_language_from_git(
            "https://github.com/kubishi/ovp-lang",
            verbose=False,
        )
    except Exception as e:
        print(f"  ⚠ Could not load language: {e}")
        return False

    generator = ExampleGenerator(agent, lang)

    # Test example generation
    print("\n  Generating examples...")
    try:
        example_set = generator.generate_examples(
            count=3,
            quality=ExampleQuality.simple,
        )
        print(f"  ✓ Generated {len(example_set.examples)} examples:")
        for ex in example_set.examples:
            print(f"    '{ex.english}' → '{ex.target}'")
            print(f"      Type: {ex.sentence_type}")
    except Exception as e:
        print(f"  ⚠ Example generation failed: {e}")

    # Test diversity analysis
    print("\n  Analyzing example diversity...")
    try:
        analysis = generator.analyze_example_diversity()
        print(f"  ✓ Analysis: {analysis.total_examples} examples")
        if analysis.gaps:
            print(f"    Gaps: {', '.join(analysis.gaps[:3])}")
        if analysis.suggestions:
            print(f"    Suggestions: {', '.join(analysis.suggestions[:2])}")
    except Exception as e:
        print(f"  ⚠ Diversity analysis failed: {e}")

    return True


def test_live_validation(agent):
    """Test live validation feedback."""
    print("\n" + "=" * 60)
    print("Testing LIVE validation feedback...")
    print("=" * 60)

    feedback = ValidationFeedback(agent)

    # Test code review
    print("\n  Reviewing sample code...")
    sample_code = '''
from yaduha.language import Sentence

LANGUAGE_NAME = "Test"
# Missing LANGUAGE_CODE!

class MySentence(Sentence["MySentence"]):
    word: str

    def __str__(self):
        return self.word

    # Missing get_examples()!

SENTENCE_TYPES = (MySentence,)
'''

    try:
        report = feedback.review_code(sample_code)
        print(f"  ✓ Review complete:")
        print(f"    Valid: {report.is_valid}")
        print(f"    Issues: {len(report.issues)}")
        for issue in report.issues[:3]:
            print(f"      [{issue.severity.value}] {issue.message[:50]}...")
        print(f"    Summary: {report.summary[:80]}...")
    except Exception as e:
        print(f"  ⚠ Code review failed: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test the language editor")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live LLM tests (requires API key)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="anthropic",
        help="LLM provider to use for live tests",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Language Editor Test Suite")
    print("=" * 60)

    # Create temp directory for tests
    temp_dir = Path(tempfile.mkdtemp(prefix="yaduha_test_"))
    print(f"\nTemp directory: {temp_dir}")

    all_passed = True

    try:
        # Run basic tests
        all_passed &= test_imports()
        all_passed &= test_data_models()
        all_passed &= test_create_language(temp_dir)
        all_passed &= test_vocab_code_generation()
        all_passed &= test_editor_with_loaded_language()

        # Run live tests if requested
        if args.live:
            print("\n" + "=" * 60)
            print(f"Running LIVE tests with {args.provider}...")
            print("=" * 60)

            try:
                agent = get_agent(args.provider)
                print(f"  Using agent: {agent.model}")

                all_passed &= test_live_vocabulary(agent)
                all_passed &= test_live_grammar(agent)
                all_passed &= test_live_examples(agent)
                all_passed &= test_live_validation(agent)

            except Exception as e:
                print(f"\n  ⚠ Could not initialize agent: {e}")
                print("  Make sure you have the appropriate API key set.")
                all_passed = False

    finally:
        # Cleanup
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
