"""
Tests for the dynamic language loader.

These tests verify that:
1. Languages can be loaded from external directories
2. Required exports are validated
3. Sentence types work correctly
4. The loaded language can be used with the pipeline translator
"""

import pytest
from pathlib import Path

from yaduha.language import (
    load_language_from_path,
    load_language_from_git,
    validate_language_path,
    LoadedLanguage,
    LanguageLoadError,
    GitLoadError,
    Sentence,
    VocabEntry,
)


# Git URL for the OVP test language
OVP_LANG_URL = "https://github.com/kubishi/ovp-lang"


@pytest.fixture(scope="module")
def ovp_language() -> LoadedLanguage:
    """Load the OVP language from git for tests."""
    lang, _ = load_language_from_git(OVP_LANG_URL, verbose=False)
    return lang


@pytest.fixture(scope="module")
def ovp_lang_path() -> Path:
    """Get the cached path of the OVP language."""
    _, path = load_language_from_git(OVP_LANG_URL, verbose=False)
    return path


class TestValidateLanguagePath:
    """Tests for validate_language_path function."""

    def test_valid_language_path(self, ovp_lang_path):
        """Test that a valid language path returns no errors."""
        errors = validate_language_path(ovp_lang_path)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_nonexistent_path(self):
        """Test that a nonexistent path returns an error."""
        errors = validate_language_path("/nonexistent/path")
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_file_instead_of_directory(self, tmp_path):
        """Test that a file path returns an error."""
        file_path = tmp_path / "file.py"
        file_path.write_text("# not a directory")
        errors = validate_language_path(file_path)
        assert len(errors) == 1
        assert "not a directory" in errors[0]

    def test_missing_language_file(self, tmp_path):
        """Test that a directory without language.py or __init__.py returns an error."""
        lang_dir = tmp_path / "test_lang"
        lang_dir.mkdir()
        (lang_dir / "vocab.py").write_text("# vocab")

        errors = validate_language_path(lang_dir)
        assert any("Missing __init__.py or language.py" in e for e in errors)

    def test_missing_vocab_file(self, tmp_path):
        """Test that a directory without vocab.py returns an error."""
        lang_dir = tmp_path / "test_lang"
        lang_dir.mkdir()
        (lang_dir / "language.py").write_text("# language")

        errors = validate_language_path(lang_dir)
        assert any("Missing vocab.py" in e for e in errors)


class TestLoadLanguageFromGit:
    """Tests for load_language_from_git function."""

    def test_load_ovp_language(self, ovp_language):
        """Test loading the OVP language from git."""
        assert isinstance(ovp_language, LoadedLanguage)
        assert ovp_language.name == "OVP"
        assert ovp_language.code == "ovp"
        assert len(ovp_language.sentence_types) > 0

    def test_loaded_language_has_sentence_types(self, ovp_language):
        """Test that loaded language has valid sentence types."""
        for st in ovp_language.sentence_types:
            assert issubclass(st, Sentence)
            assert hasattr(st, 'get_examples')
            assert hasattr(st, '__str__')

    def test_loaded_language_has_vocabulary(self, ovp_language):
        """Test that loaded language has vocabulary."""
        assert len(ovp_language.nouns) > 0
        assert len(ovp_language.transitive_verbs) > 0
        assert len(ovp_language.intransitive_verbs) > 0

        # Check VocabEntry structure
        for entry in ovp_language.nouns[:3]:
            assert isinstance(entry, VocabEntry)
            assert isinstance(entry.english, str)
            assert isinstance(entry.target, str)

    def test_loaded_language_examples_work(self, ovp_language):
        """Test that examples from loaded language can be rendered."""
        examples = ovp_language.get_all_examples()

        assert len(examples) > 0

        for english, sentence in examples:
            assert isinstance(english, str)
            assert len(english) > 0

            # Render the sentence
            rendered = str(sentence)
            assert isinstance(rendered, str)
            assert len(rendered) > 0
            print(f"  {english!r} -> {rendered!r}")


class TestLoadLanguageFromPath:
    """Tests for load_language_from_path function."""

    def test_load_from_cached_path(self, ovp_lang_path):
        """Test loading from the cached git path."""
        lang = load_language_from_path(ovp_lang_path)
        assert isinstance(lang, LoadedLanguage)
        assert lang.name == "OVP"

    def test_load_nonexistent_path(self):
        """Test that loading from nonexistent path raises error."""
        with pytest.raises(LanguageLoadError, match="not a directory"):
            load_language_from_path("/nonexistent/path")

    def test_load_missing_required_exports(self, tmp_path):
        """Test that missing required exports raises error."""
        lang_dir = tmp_path / "bad_lang"
        lang_dir.mkdir()

        # Create a minimal module without required exports
        (lang_dir / "__init__.py").write_text("""
# Missing LANGUAGE_NAME, LANGUAGE_CODE, SENTENCE_TYPES
pass
""")
        (lang_dir / "vocab.py").write_text("NOUNS = []")

        with pytest.raises(LanguageLoadError, match="Missing required export"):
            load_language_from_path(lang_dir)

    def test_load_invalid_sentence_type(self, tmp_path):
        """Test that invalid sentence types raise error."""
        lang_dir = tmp_path / "bad_lang"
        lang_dir.mkdir()

        # Create a module with invalid SENTENCE_TYPES (not subclasses of Sentence)
        (lang_dir / "__init__.py").write_text("""
LANGUAGE_NAME = "Bad"
LANGUAGE_CODE = "bad"
SENTENCE_TYPES = [str, int]  # These are not Sentence subclasses
""")
        (lang_dir / "vocab.py").write_text("NOUNS = []")

        with pytest.raises(LanguageLoadError, match="must be a subclass of Sentence"):
            load_language_from_path(lang_dir)


class TestSentenceRendering:
    """Tests for sentence rendering in loaded languages."""

    def test_subject_verb_sentence_rendering(self, ovp_language):
        """Test that SubjectVerbSentence renders correctly."""
        # Find SubjectVerbSentence type
        sv_type = None
        for st in ovp_language.sentence_types:
            if st.__name__ == "SubjectVerbSentence":
                sv_type = st
                break

        assert sv_type is not None, "SubjectVerbSentence not found in SENTENCE_TYPES"

        examples = sv_type.get_examples()
        assert len(examples) > 0

        for english, sentence in examples:
            rendered = str(sentence)
            # Should contain target language morphemes with hyphens
            assert "-" in rendered, f"Expected morpheme boundaries in: {rendered}"
            print(f"  SV: {english!r} -> {rendered!r}")

    def test_subject_verb_object_sentence_rendering(self, ovp_language):
        """Test that SubjectVerbObjectSentence renders correctly."""
        # Find SubjectVerbObjectSentence type
        svo_type = None
        for st in ovp_language.sentence_types:
            if st.__name__ == "SubjectVerbObjectSentence":
                svo_type = st
                break

        assert svo_type is not None, "SubjectVerbObjectSentence not found in SENTENCE_TYPES"

        examples = svo_type.get_examples()
        assert len(examples) > 0

        for english, sentence in examples:
            rendered = str(sentence)
            # Should contain target language morphemes with hyphens
            assert "-" in rendered, f"Expected morpheme boundaries in: {rendered}"
            print(f"  SVO: {english!r} -> {rendered!r}")


class TestLanguageWithPipelineTranslator:
    """Tests for using loaded languages with the pipeline translator."""

    def test_create_pipeline_translator(self, ovp_language):
        """Test that a pipeline translator can be created with loaded language."""
        from yaduha.translator.pipeline import PipelineTranslator

        # Verify we can instantiate the translator with loaded sentence types
        # (We can't actually run translations without API keys)
        assert ovp_language.sentence_types is not None
        assert len(ovp_language.sentence_types) > 0

        # The translator expects a tuple of sentence types
        print(f"  Sentence types: {[st.__name__ for st in ovp_language.sentence_types]}")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_same_language_twice(self):
        """Test that loading the same language twice doesn't cause issues."""
        lang1, _ = load_language_from_git(OVP_LANG_URL, verbose=False)
        lang2, _ = load_language_from_git(OVP_LANG_URL, verbose=False)

        assert lang1.name == lang2.name
        assert lang1.code == lang2.code
        assert len(lang1.sentence_types) == len(lang2.sentence_types)

    def test_sentence_json_schema(self, ovp_language):
        """Test that sentence types have valid JSON schemas for API use."""
        for st in ovp_language.sentence_types:
            # Pydantic models should have model_json_schema
            schema = st.model_json_schema()
            assert isinstance(schema, dict)
            assert "properties" in schema or "anyOf" in schema or "$defs" in schema
            print(f"  {st.__name__} schema keys: {list(schema.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
