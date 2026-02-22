"""Pydantic models for API request and response bodies."""

from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for which agent/model to use."""

    provider: str = Field(..., description="Agent provider: openai, anthropic, gemini, ollama")
    model: str = Field(..., description="Model name (e.g., gpt-4o, claude-sonnet-4-5)")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


# -- Language responses --


class SentenceTypeInfo(BaseModel):
    name: str
    field_count: int


class LanguageSummary(BaseModel):
    code: str
    name: str
    sentence_type_count: int
    sentence_types: list[str]


class LanguageDetail(BaseModel):
    code: str
    name: str
    sentence_types: list[SentenceTypeInfo]


class LanguageListResponse(BaseModel):
    languages: list[LanguageSummary]


# -- Schema / example responses --


class SentenceSchemaResponse(BaseModel):
    language_code: str
    sentence_type: str
    json_schema: dict[str, Any]


class ExamplePair(BaseModel):
    english: str
    structured: dict[str, Any]
    rendered: str


class SentenceExamplesResponse(BaseModel):
    language_code: str
    sentence_type: str
    examples: list[ExamplePair]


class RenderResponse(BaseModel):
    language_code: str
    sentence_type: str
    rendered: str
    structured: dict[str, Any]


class ToEnglishRequest(BaseModel):
    data: dict[str, Any] = Field(
        ..., description="Structured sentence data matching the sentence type schema"
    )
    agent: "AgentConfig"


class ToEnglishResponse(BaseModel):
    language_code: str
    sentence_type: str
    rendered: str
    english: str
    structured: dict[str, Any]


# -- Evaluator config --


class EvaluatorConfig(BaseModel):
    """Configuration for a translation quality evaluator."""

    type: str = Field(
        ..., description="Evaluator type: 'openai_embedding', 'chrf', 'bleu', 'bertscore', 'comet'"
    )
    model: str | None = Field(
        default=None,
        description="Model name for the evaluator (e.g., 'text-embedding-3-small')",
    )


# -- Translation requests --


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="English text to translate")
    language_code: str = Field(..., description="Target language code (e.g., 'ovp')")
    agent: AgentConfig
    back_translation_agent: AgentConfig | None = None
    evaluators: list[EvaluatorConfig] | None = None


class AgenticTranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language_code: str
    agent: AgentConfig
    system_prompt: str | None = None
