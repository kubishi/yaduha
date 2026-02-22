"""Routes for sentence type schemas, examples, and rendering."""

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request, status
from pydantic import ValidationError

from yaduha.api.dependencies import create_agent, get_language, get_sentence_type
from yaduha.api.models import (
    ExamplePair,
    RenderResponse,
    SentenceExamplesResponse,
    SentenceSchemaResponse,
    ToEnglishRequest,
    ToEnglishResponse,
)
from yaduha.tool.sentence_to_english import SentenceToEnglishTool

router = APIRouter(prefix="/languages/{language_code}/sentence-types", tags=["schemas"])


@router.get("", response_model=list[SentenceSchemaResponse])
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


@router.post("/{sentence_type_name}/render", response_model=RenderResponse)
async def render_sentence(
    language_code: str,
    sentence_type_name: str,
    data: dict[str, Any] = Body(...),
):
    """Render a structured sentence in the target language.

    Accepts a JSON body matching the sentence type's schema and returns
    the rendered target-language string.
    """
    lang = get_language(language_code)
    st = get_sentence_type(lang, sentence_type_name)
    try:
        instance = st.model_validate(data)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in e.errors()
            ],
        )
    return RenderResponse(
        language_code=language_code,
        sentence_type=sentence_type_name,
        rendered=str(instance),
        structured=instance.model_dump(),
    )


def _headers_dict(request: Request) -> dict[str, str]:
    return {k.lower(): v for k, v in request.headers.items()}


@router.post("/{sentence_type_name}/to-english", response_model=ToEnglishResponse)
async def sentence_to_english(
    language_code: str,
    sentence_type_name: str,
    body: ToEnglishRequest,
    request: Request,
):
    """Translate a structured sentence to English.

    Accepts structured sentence data and an agent config. Uses the agent to
    translate the sentence to natural English.
    """
    lang = get_language(language_code)
    st = get_sentence_type(lang, sentence_type_name)
    try:
        instance = st.model_validate(body.data)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in e.errors()
            ],
        )

    headers = _headers_dict(request)
    agent = create_agent(body.agent, headers)

    tool = SentenceToEnglishTool(agent=agent, SentenceType=lang.sentence_types)
    response = tool(instance)

    return ToEnglishResponse(
        language_code=language_code,
        sentence_type=sentence_type_name,
        rendered=str(instance),
        english=response.content,
        structured=instance.model_dump(),
    )
