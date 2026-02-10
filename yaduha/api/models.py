"""Pydantic models for API request and response bodies."""

from typing import Any, Dict, List, Optional

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
    sentence_types: List[str]


class LanguageDetail(BaseModel):
    code: str
    name: str
    sentence_types: List[SentenceTypeInfo]


class LanguageListResponse(BaseModel):
    languages: List[LanguageSummary]


# -- Schema / example responses --


class SentenceSchemaResponse(BaseModel):
    language_code: str
    sentence_type: str
    json_schema: Dict[str, Any]


class ExamplePair(BaseModel):
    english: str
    structured: Dict[str, Any]
    rendered: str


class SentenceExamplesResponse(BaseModel):
    language_code: str
    sentence_type: str
    examples: List[ExamplePair]


# -- Translation requests --


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="English text to translate")
    language_code: str = Field(..., description="Target language code (e.g., 'ovp')")
    agent: AgentConfig
    back_translation_agent: Optional[AgentConfig] = None


class AgenticTranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language_code: str
    agent: AgentConfig
    system_prompt: Optional[str] = None
