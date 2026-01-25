"""
Integration tests for using loaded languages with the pipeline translator.

These tests verify end-to-end translation functionality with dynamically loaded languages.
Some tests require API keys and are skipped if not available.
"""

import os
import pytest

from yaduha.language import load_language_from_git, LoadedLanguage
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.translator import Translation

# Git URL for the OVP test language
OVP_LANG_URL = "https://github.com/kubishi/ovp-lang"


def has_anthropic_key() -> bool:
    """Check if Anthropic API key is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture(scope="module")
def ovp_language() -> LoadedLanguage:
    """Load the OVP language from git for tests."""
    lang, _ = load_language_from_git(OVP_LANG_URL, verbose=False)
    return lang


class TestPipelineTranslatorWithLoadedLanguage:
    """Tests for PipelineTranslator using dynamically loaded languages."""

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_translate_simple_sentence_anthropic(self, ovp_language):
        """Test translating a simple sentence using Anthropic."""
        from yaduha.agent.anthropic import AnthropicAgent

        agent = AnthropicAgent(
            model="claude-sonnet-4-20250514",
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

        translator = PipelineTranslator(
            agent=agent,
            SentenceType=ovp_language.sentence_types,
        )

        result = translator.translate("I sleep.")

        assert isinstance(result, Translation)
        assert result.source == "I sleep."
        assert len(result.target) > 0
        assert "-" in result.target  # OVP uses morpheme boundaries

        print(f"\n  Source: {result.source}")
        print(f"  Target: {result.target}")
        if result.back_translation:
            print(f"  Back translation: {result.back_translation.source}")

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_translate_complex_sentence_anthropic(self, ovp_language):
        """Test translating a more complex sentence using Anthropic."""
        from yaduha.agent.anthropic import AnthropicAgent

        agent = AnthropicAgent(
            model="claude-sonnet-4-20250514",
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

        translator = PipelineTranslator(
            agent=agent,
            SentenceType=ovp_language.sentence_types,
        )

        result = translator.translate("The coyote eats the fish.")

        assert isinstance(result, Translation)
        assert len(result.target) > 0

        print(f"\n  Source: {result.source}")
        print(f"  Target: {result.target}")
        if result.back_translation:
            print(f"  Back translation: {result.back_translation.source}")

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_translate_simple_sentence_openai(self, ovp_language):
        """Test translating a simple sentence using OpenAI."""
        from yaduha.agent.openai import OpenAIAgent

        agent = OpenAIAgent(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

        translator = PipelineTranslator(
            agent=agent,
            SentenceType=ovp_language.sentence_types,
        )

        result = translator.translate("The dog runs.")

        assert isinstance(result, Translation)
        assert len(result.target) > 0

        print(f"\n  Source: {result.source}")
        print(f"  Target: {result.target}")


class TestTranslatorExamples:
    """Test that translator examples work with loaded languages."""

    def test_get_examples_from_translator(self, ovp_language):
        """Test that translator can generate examples from loaded language."""
        # We can test the example generation without API calls
        for st in ovp_language.sentence_types:
            examples = st.get_examples()
            assert len(examples) > 0

            for english, sentence in examples:
                # Verify the sentence can be rendered
                target = str(sentence)
                assert len(target) > 0

                # Verify JSON serialization works (needed for API)
                json_data = sentence.model_dump()
                assert isinstance(json_data, dict)

                # Verify we can reconstruct from JSON
                reconstructed = st.model_validate(json_data)
                assert str(reconstructed) == target

                print(f"  {st.__name__}: {english!r} -> {target!r}")


class TestVocabularyIntegration:
    """Tests for vocabulary usage in translation."""

    def test_vocabulary_in_rendered_output(self, ovp_language):
        """Test that vocabulary words appear in rendered sentences."""
        # Check that examples use vocabulary
        examples = ovp_language.get_all_examples()

        for _, sentence in examples:
            rendered = str(sentence)
            # Just verify we're producing output
            assert len(rendered) > 0

    def test_all_nouns_have_target(self, ovp_language):
        """Test that all nouns have target translations."""
        for noun in ovp_language.nouns:
            assert noun.english, "Noun must have English"
            assert noun.target, f"Noun '{noun.english}' must have target"

    def test_all_verbs_have_target(self, ovp_language):
        """Test that all verbs have target translations."""
        all_verbs = ovp_language.transitive_verbs + ovp_language.intransitive_verbs
        for verb in all_verbs:
            assert verb.english, "Verb must have English"
            assert verb.target, f"Verb '{verb.english}' must have target"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
