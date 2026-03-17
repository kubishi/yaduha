"""Tests for yaduha.experiments: models, validation, and run_experiment()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tests.conftest import FakeAgent, SimpleSentence
from yaduha.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ModelConfig,
    ProviderConfig,
    SentenceInput,
    SentenceResult,
    run_experiment,
)
from yaduha.language import Language
from yaduha.translator import Translation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_translation(target: str = "translated") -> Translation:
    return Translation(
        source="input",
        target=target,
        translation_time=0.1,
        prompt_tokens=10,
        completion_tokens=5,
    )


def _make_language() -> Language:
    return Language(code="test", name="Test Language", sentence_types=(SimpleSentence,))


def _minimal_config(tmp_path, **overrides) -> ExperimentConfig:
    defaults: dict = dict(
        name="test",
        language_code="test",
        providers=[
            ProviderConfig(
                provider="openai",
                models=[ModelConfig(model="gpt-4o-mini")],
                api_key="sk-test",
            )
        ],
        sentences=[SentenceInput(text="The dog runs.")],
        savedir=tmp_path,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# SentenceInput validation
# ---------------------------------------------------------------------------


def test_sentence_input_defaults():
    s = SentenceInput(text="Hello world.")
    assert s.text == "Hello world."
    assert s.metadata == {}


def test_sentence_input_with_metadata():
    s = SentenceInput(
        text="The dog runs.",
        metadata={"sentence_type": "simple", "difficulty": "easy"},
    )
    assert s.metadata["sentence_type"] == "simple"
    assert s.metadata["difficulty"] == "easy"


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


def test_model_config_defaults():
    m = ModelConfig(model="gpt-4o-mini")
    assert m.temperature == 0.0


def test_model_config_temperature_too_high():
    with pytest.raises(ValidationError):
        ModelConfig(model="gpt-4o-mini", temperature=3.0)


def test_model_config_temperature_negative():
    with pytest.raises(ValidationError):
        ModelConfig(model="gpt-4o-mini", temperature=-0.1)


# ---------------------------------------------------------------------------
# ProviderConfig validation
# ---------------------------------------------------------------------------


def test_provider_config_requires_at_least_one_model():
    with pytest.raises(ValidationError):
        ProviderConfig(provider="openai", models=[])


def test_provider_config_multiple_models():
    p = ProviderConfig(
        provider="openai",
        models=[ModelConfig(model="gpt-4o-mini"), ModelConfig(model="gpt-4o")],
    )
    assert len(p.models) == 2


def test_provider_config_api_key_optional():
    p = ProviderConfig(provider="ollama", models=[ModelConfig(model="llama3")])
    assert p.api_key is None


# -- ExperimentConfig validation --


def test_experiment_config_requires_sentences():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="test",
            language_code="ovp",
            providers=[
                ProviderConfig(provider="openai", models=[ModelConfig(model="gpt-4o-mini")])
            ],
            sentences=[],
        )


def test_experiment_config_requires_providers():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="test",
            language_code="ovp",
            providers=[],
            sentences=[SentenceInput(text="Hello.")],
        )


def test_experiment_config_defaults():
    config = ExperimentConfig(
        name="test-exp",
        language_code="ovp",
        providers=[
            ProviderConfig(
                provider="openai",
                models=[ModelConfig(model="gpt-4o-mini")],
            )
        ],
        sentences=[SentenceInput(text="The dog runs.")],
    )
    assert config.translator_type == "pipeline"
    assert config.savedir is None  # uses RESULTS_DIR at runtime


def test_experiment_config_multi_provider_multi_model():
    config = ExperimentConfig(
        name="multi",
        language_code="ovp",
        providers=[
            ProviderConfig(
                provider="openai",
                models=[ModelConfig(model="gpt-4o-mini"), ModelConfig(model="gpt-4o")],
            ),
            ProviderConfig(
                provider="anthropic",
                models=[ModelConfig(model="claude-haiku-4-5-20251001")],
                api_key="sk-ant-test",
            ),
        ],
        sentences=[
            SentenceInput(text="The dog runs."),
            SentenceInput(text="I see the mountains.", metadata={"sentence_type": "transitive"}),
        ],
    )
    assert len(config.providers) == 2
    assert len(config.sentences) == 2
    assert config.sentences[1].metadata["sentence_type"] == "transitive"


# -- run_experiment success path --


@patch("yaduha.experiments.LanguageLoader")
@patch("yaduha.experiments._create_agent")
def test_run_experiment_success(mock_create_agent, mock_loader, tmp_path):
    mock_loader.load_language.return_value = _make_language()

    agent = FakeAgent()
    agent.set_response(SimpleSentence(subject="nüü", verb="üwi"))
    mock_create_agent.return_value = agent

    mock_translator = MagicMock()
    mock_translator.translate.return_value = _make_translation("translated!")

    config = _minimal_config(tmp_path)

    with patch("yaduha.experiments.PipelineTranslator", return_value=mock_translator):
        result = run_experiment(config)

    assert isinstance(result, ExperimentResult)
    assert result.name == "test"
    assert result.filename == "test.jsonl"
    assert len(result.results) == 1
    r = result.results[0]
    assert r.provider == "openai"
    assert r.model == "gpt-4o-mini"
    assert r.translation is not None
    assert r.error is None


@patch("yaduha.experiments.LanguageLoader")
@patch("yaduha.experiments._create_agent")
def test_run_experiment_writes_jsonl(mock_create_agent, mock_loader, tmp_path):
    mock_loader.load_language.return_value = _make_language()

    agent = FakeAgent()
    agent.set_response(SimpleSentence(subject="nüü", verb="üwi"))
    mock_create_agent.return_value = agent

    mock_translator = MagicMock()
    mock_translator.translate.return_value = _make_translation()

    config = _minimal_config(tmp_path)

    with patch("yaduha.experiments.PipelineTranslator", return_value=mock_translator):
        result = run_experiment(config)

    assert result.filename == "test.jsonl"
    assert (tmp_path / result.filename).parent == tmp_path


# run_experiment — failure paths


@patch("yaduha.experiments.LanguageLoader")
@patch("yaduha.experiments._create_agent")
def test_run_experiment_agent_creation_failure(mock_create_agent, mock_loader, tmp_path):
    mock_loader.load_language.return_value = _make_language()
    mock_create_agent.side_effect = ValueError("Invalid API key")

    result = run_experiment(_minimal_config(tmp_path))

    assert len(result.results) == 1
    assert result.results[0].error == "Invalid API key"
    assert result.results[0].translation is None


@patch("yaduha.experiments.LanguageLoader")
@patch("yaduha.experiments._create_agent")
def test_run_experiment_translation_failure(mock_create_agent, mock_loader, tmp_path):
    mock_loader.load_language.return_value = _make_language()

    agent = FakeAgent()
    mock_create_agent.return_value = agent

    mock_translator = MagicMock()
    mock_translator.translate.side_effect = RuntimeError("API timeout")

    config = _minimal_config(tmp_path)

    with patch("yaduha.experiments.PipelineTranslator", return_value=mock_translator):
        result = run_experiment(config)

    assert result.results[0].error == "API timeout"
    assert result.results[0].translation is None


@patch("yaduha.experiments.LanguageLoader")
def test_run_experiment_unknown_language(mock_loader, tmp_path):
    from yaduha.language.exceptions import LanguageNotFoundError

    mock_loader.load_language.side_effect = LanguageNotFoundError("xyz")

    config = _minimal_config(tmp_path, language_code="xyz")

    with pytest.raises(ValueError, match="Language 'xyz' not found"):
        run_experiment(config)


def test_run_experiment_file_exists_no_overwrite(tmp_path):
    existing = tmp_path / "test.jsonl"
    existing.write_text('{"event": "old"}\n')

    config = _minimal_config(tmp_path)

    with pytest.raises(FileExistsError):
        run_experiment(config, overwrite=False)


# run_experiment — multi-provider / multi-model combinations


@patch("yaduha.experiments.LanguageLoader")
@patch("yaduha.experiments._create_agent")
def test_run_experiment_multi_provider_multi_model(mock_create_agent, mock_loader, tmp_path):
    mock_loader.load_language.return_value = _make_language()

    agent = FakeAgent()
    mock_create_agent.return_value = agent

    mock_translator = MagicMock()
    mock_translator.translate.return_value = _make_translation()

    config = ExperimentConfig(
        name="multi",
        language_code="test",
        providers=[
            ProviderConfig(
                provider="openai",
                models=[ModelConfig(model="gpt-4o-mini"), ModelConfig(model="gpt-4o")],
                api_key="sk-test",
            ),
            ProviderConfig(
                provider="anthropic",
                models=[ModelConfig(model="claude-haiku-4-5-20251001")],
                api_key="sk-ant-test",
            ),
        ],
        sentences=[
            SentenceInput(text="The dog runs."),
            SentenceInput(text="I see mountains."),
        ],
        savedir=tmp_path,
    )

    with patch("yaduha.experiments.PipelineTranslator", return_value=mock_translator):
        result = run_experiment(config)

    # 3 provider/model combos × 2 sentences = 6 results
    assert len(result.results) == 6
    providers = {r.provider for r in result.results}
    assert providers == {"openai", "anthropic"}
    models = {r.model for r in result.results}
    assert models == {"gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001"}


@patch("yaduha.experiments.LanguageLoader")
@patch("yaduha.experiments._create_agent")
def test_run_experiment_partial_failure(mock_create_agent, mock_loader, tmp_path):
    """One provider fails agent creation; the other succeeds."""
    mock_loader.load_language.return_value = _make_language()

    agent = FakeAgent()

    def _side_effect(provider, model, temperature, api_key):
        if provider == "openai":
            return agent
        raise ValueError("Anthropic key missing")

    mock_create_agent.side_effect = _side_effect

    mock_translator = MagicMock()
    mock_translator.translate.return_value = _make_translation()

    config = ExperimentConfig(
        name="partial",
        language_code="test",
        providers=[
            ProviderConfig(
                provider="openai",
                models=[ModelConfig(model="gpt-4o-mini")],
                api_key="sk-test",
            ),
            ProviderConfig(
                provider="anthropic",
                models=[ModelConfig(model="claude-haiku-4-5-20251001")],
            ),
        ],
        sentences=[SentenceInput(text="The dog runs.")],
        savedir=tmp_path,
    )

    with patch("yaduha.experiments.PipelineTranslator", return_value=mock_translator):
        result = run_experiment(config)

    assert len(result.results) == 2
    openai_r = next(r for r in result.results if r.provider == "openai")
    anthropic_r = next(r for r in result.results if r.provider == "anthropic")
    assert openai_r.translation is not None
    assert anthropic_r.error == "Anthropic key missing"


# ---------------------------------------------------------------------------
# SentenceResult model
# ---------------------------------------------------------------------------


def test_sentence_result_requires_sentence():
    with pytest.raises(ValidationError):
        SentenceResult(provider="openai", model="gpt-4o-mini")  # type: ignore[call-arg]


def test_sentence_result_optional_translation_and_error():
    r = SentenceResult(
        sentence=SentenceInput(text="Hello."),
        provider="openai",
        model="gpt-4o-mini",
    )
    assert r.translation is None
    assert r.error is None
