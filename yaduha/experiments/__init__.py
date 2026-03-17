"""Experiment framework for running batch translations across providers, models, and sentences."""

from __future__ import annotations

import pathlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from yaduha.language.exceptions import LanguageNotFoundError
from yaduha.loader import LanguageLoader
from yaduha.logger import JsonLogger
from yaduha.tool.english_to_sentences import EnglishToSentencesTool
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha.translator import Translation
from yaduha.translator.agentic import AgenticTranslator
from yaduha.translator.pipeline import PipelineTranslator

# Always points to the repo-root results/ directory, regardless of cwd.
RESULTS_DIR = pathlib.Path(__file__).parent.parent.parent / "results"

# Inputs


class SentenceInput(BaseModel):
    """A sentence to translate, with optional metadata."""

    text: str = Field(..., description="English sentence to translate.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional metadata attached to this sentence (e.g. sentence_type, difficulty, notes)."
        ),
    )


class ModelConfig(BaseModel):
    """Configuration for a single model within a provider."""

    model: str = Field(..., description="Model identifier (e.g. 'gpt-4o-mini').")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ProviderConfig(BaseModel):
    """One LLM provider with one or more models to evaluate."""

    provider: str = Field(..., description="Provider name: openai, anthropic, gemini, ollama")
    models: list[ModelConfig] = Field(..., min_length=1)
    api_key: str | None = Field(
        default=None,
        description="API key. Falls back to the matching environment variable if omitted.",
    )


class ExperimentConfig(BaseModel):
    """Full configuration for a batch translation experiment.

    Runs every (provider, model) × sentence combination and logs everything
    to a single JSONL file in ``savedir`` (defaults to ``RESULTS_DIR``).
    """

    name: str = Field(..., description="Experiment name, used as the JSONL log filename.")
    language_code: str = Field(..., description="Target language code (e.g. 'ovp').")
    translator_type: Literal["pipeline", "agentic"] = Field(default="pipeline")
    providers: list[ProviderConfig] = Field(..., min_length=1)
    sentences: list[SentenceInput] = Field(..., min_length=1)
    savedir: pathlib.Path | None = Field(
        default=None,
        description="Directory to write JSONL logs. Defaults to RESULTS_DIR.",
    )


# Output/results models


class SentenceResult(BaseModel):
    """Result for one sentence translated by one (provider, model) pair."""

    sentence: SentenceInput
    provider: str
    model: str
    translation: Translation | None = None
    error: str | None = None


class ExperimentResult(BaseModel):
    """Aggregated results for a complete experiment run."""

    name: str
    filename: str
    results: list[SentenceResult]


# Agents


def _create_agent(provider: str, model: str, temperature: float, api_key: str | None):
    """Instantiate an agent for the given provider and model.

    Raises:
        ValueError: Unknown provider or missing API key.
    """
    import importlib
    import os

    _REGISTRY: dict[str, tuple[str, str, str | None, bool]] = {
        # provider -> (module, class, env_var, requires_key)
        "openai": ("yaduha.agent.openai", "OpenAIAgent", "OPENAI_API_KEY", True),
        "anthropic": ("yaduha.agent.anthropic", "AnthropicAgent", "ANTHROPIC_API_KEY", True),
        "gemini": ("yaduha.agent.gemini", "GeminiAgent", "GEMINI_API_KEY", True),
        "ollama": ("yaduha.agent.ollama", "OllamaAgent", None, False),
    }

    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(_REGISTRY)}")

    module_path, cls_name, env_var, requires_key = _REGISTRY[provider]
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    if requires_key:
        key = api_key or (os.environ.get(env_var) if env_var else None)
        if not key:
            raise ValueError(
                f"API key required for '{provider}'. "
                f"Pass api_key in ProviderConfig or set {env_var}."
            )
        return cls(model=model, api_key=key, temperature=temperature)
    else:
        return cls(model=model, temperature=temperature)


# Experiment runner


def run_experiment(config: ExperimentConfig, overwrite: bool = True) -> ExperimentResult:
    """Run a batch translation experiment across all provider/model/sentence combinations.

    For every (provider, model) pair in ``config.providers``, each sentence in
    ``config.sentences`` is translated and all agent events are logged to a single
    JSONL file (``<savedir>/<name>.jsonl``).

    Args:
        config: Full experiment configuration.
        overwrite: Overwrite an existing log file. Default ``True``. Pass
            ``False`` to raise ``FileExistsError`` if the file already exists.

    Returns:
        :class:`ExperimentResult` containing all translation outcomes.

    Raises:
        ValueError: Language code not found.
        FileExistsError: Log file exists and ``overwrite=False``.
    """
    savedir = config.savedir or RESULTS_DIR
    savedir.mkdir(parents=True, exist_ok=True)

    filename = f"{config.name}.jsonl"
    log_path = savedir / filename

    if log_path.exists():
        if overwrite:
            log_path.unlink()
        else:
            raise FileExistsError(
                f"Log file already exists: {log_path}. Pass overwrite=True to replace it."
            )

    try:
        language = LanguageLoader.load_language(config.language_code)
    except LanguageNotFoundError:
        raise ValueError(f"Language '{config.language_code}' not found.")

    logger = JsonLogger(file_path=log_path)
    all_results: list[SentenceResult] = []

    for prov_cfg in config.providers:
        for model_cfg in prov_cfg.models:
            # --- build agent ---
            try:
                agent = _create_agent(
                    prov_cfg.provider,
                    model_cfg.model,
                    model_cfg.temperature,
                    prov_cfg.api_key,
                )
            except Exception as e:
                # Record failure for every sentence under this provider/model
                for sentence in config.sentences:
                    all_results.append(
                        SentenceResult(
                            sentence=sentence,
                            provider=prov_cfg.provider,
                            model=model_cfg.model,
                            error=str(e),
                        )
                    )
                continue

            agent = agent.model_copy(update={"logger": logger})

            # --- build translator ---
            if config.translator_type == "pipeline":
                translator = PipelineTranslator(
                    agent=agent,
                    SentenceType=language.sentence_types,
                    logger=logger,
                )
            else:
                e2s = EnglishToSentencesTool(
                    agent=agent, SentenceType=language.sentence_types, logger=logger
                )
                s2e = SentenceToEnglishTool(
                    agent=agent, SentenceType=language.sentence_types, logger=logger
                )
                pipe = PipelineTranslator(
                    agent=agent, SentenceType=language.sentence_types, logger=logger
                )
                translator = AgenticTranslator(agent=agent, tools=[e2s, s2e, pipe], logger=logger)

            # --- translate each sentence ---
            for sentence in config.sentences:
                try:
                    translation = translator.translate(sentence.text)
                    all_results.append(
                        SentenceResult(
                            sentence=sentence,
                            provider=prov_cfg.provider,
                            model=model_cfg.model,
                            translation=translation,
                        )
                    )
                except Exception as e:
                    all_results.append(
                        SentenceResult(
                            sentence=sentence,
                            provider=prov_cfg.provider,
                            model=model_cfg.model,
                            error=str(e),
                        )
                    )

    return ExperimentResult(name=config.name, filename=filename, results=all_results)
