"""FastAPI dependency injection: agent factory, API key resolution, language lookup."""

import os
from typing import Optional

from fastapi import HTTPException, status

from yaduha.agent.openai import OpenAIAgent
from yaduha.agent.anthropic import AnthropicAgent
from yaduha.agent.gemini import GeminiAgent
from yaduha.agent.ollama import OllamaAgent
from yaduha.language.language import Language
from yaduha.language.exceptions import LanguageNotFoundError
from yaduha.loader import LanguageLoader

from yaduha.api.models import AgentConfig

_PROVIDERS = {
    "openai": {
        "cls": OpenAIAgent,
        "env_var": "OPENAI_API_KEY",
        "header": "x-openai-key",
        "requires_key": True,
    },
    "anthropic": {
        "cls": AnthropicAgent,
        "env_var": "ANTHROPIC_API_KEY",
        "header": "x-anthropic-key",
        "requires_key": True,
    },
    "gemini": {
        "cls": GeminiAgent,
        "env_var": "GEMINI_API_KEY",
        "header": "x-gemini-key",
        "requires_key": True,
    },
    "ollama": {
        "cls": OllamaAgent,
        "env_var": None,
        "header": None,
        "requires_key": False,
    },
}


def _resolve_api_key(provider: str, headers: dict[str, str]) -> Optional[str]:
    info = _PROVIDERS[provider]
    if not info["requires_key"]:
        return None
    # Provider-specific header
    if info["header"] and info["header"] in headers:
        return headers[info["header"]]
    # Generic header
    if "x-api-key" in headers:
        return headers["x-api-key"]
    # Environment variable
    if info["env_var"]:
        key = os.environ.get(info["env_var"])
        if key:
            return key
    return None


def create_agent(config: AgentConfig, headers: dict[str, str]):
    if config.provider not in _PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider '{config.provider}'. Available: {list(_PROVIDERS)}",
        )

    info = _PROVIDERS[config.provider]

    if info["requires_key"]:
        api_key = _resolve_api_key(config.provider, headers)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    f"No API key for provider '{config.provider}'. "
                    f"Set {info['env_var']} env var or pass '{info['header']}' header."
                ),
            )
        try:
            return info["cls"](
                model=config.model,
                api_key=api_key,
                temperature=config.temperature,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid agent configuration: {e}",
            )
    else:
        try:
            return info["cls"](model=config.model, temperature=config.temperature)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid agent configuration: {e}",
            )


def get_language(language_code: str) -> Language:
    try:
        return LanguageLoader.load_language(language_code)
    except LanguageNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Language '{language_code}' not found. Use GET /languages to list installed languages.",
        )


def get_sentence_type(language: Language, sentence_type_name: str):
    for st in language.sentence_types:
        if st.__name__ == sentence_type_name:
            return st
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"Sentence type '{sentence_type_name}' not found in language '{language.code}'. "
            f"Available: {[st.__name__ for st in language.sentence_types]}"
        ),
    )
