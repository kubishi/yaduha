"""Routes for sentence type schemas and examples."""

from typing import List

from fastapi import APIRouter

from yaduha.api.models import (
    SentenceSchemaResponse,
    SentenceExamplesResponse,
    ExamplePair,
)
from yaduha.api.dependencies import get_language, get_sentence_type

router = APIRouter(prefix="/languages/{language_code}/sentence-types", tags=["schemas"])


@router.get("", response_model=List[SentenceSchemaResponse])
async def list_sentence_type_schemas(language_code: str):
    """List all sentence type schemas for a language."""
    lang = get_language(language_code)
    return [
        SentenceSchemaResponse(
            language_code=language_code,
            sentence_type=st.__name__,
            json_schema=st.model_json_schema(),
        )
        for st in lang.sentence_types
    ]


@router.get("/{sentence_type_name}/schema", response_model=SentenceSchemaResponse)
async def get_sentence_schema(language_code: str, sentence_type_name: str):
    """Get the JSON schema for a sentence type."""
    lang = get_language(language_code)
    st = get_sentence_type(lang, sentence_type_name)
    return SentenceSchemaResponse(
        language_code=language_code,
        sentence_type=sentence_type_name,
        json_schema=st.model_json_schema(),
    )


@router.get("/{sentence_type_name}/examples", response_model=SentenceExamplesResponse)
async def get_sentence_examples(language_code: str, sentence_type_name: str):
    """Get examples for a sentence type.

    Each example includes the English source, the structured Pydantic model
    data, and the rendered target-language string.
    """
    lang = get_language(language_code)
    st = get_sentence_type(lang, sentence_type_name)
    examples = st.get_examples()
    return SentenceExamplesResponse(
        language_code=language_code,
        sentence_type=sentence_type_name,
        examples=[
            ExamplePair(
                english=english,
                structured=instance.model_dump(),
                rendered=str(instance),
            )
            for english, instance in examples
        ],
    )
