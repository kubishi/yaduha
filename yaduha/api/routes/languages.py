"""Routes for language discovery."""

from fastapi import APIRouter

from yaduha.api.dependencies import get_language
from yaduha.api.models import (
    LanguageDetail,
    LanguageListResponse,
    LanguageSummary,
    SentenceTypeInfo,
)
from yaduha.loader import LanguageLoader

router = APIRouter(prefix="/languages", tags=["languages"])


@router.get("", response_model=LanguageListResponse)
async def list_languages():
    """List all installed language packages."""
    languages = LanguageLoader.list_installed_languages()
    return LanguageListResponse(
        languages=[
            LanguageSummary(
                code=lang.code,
                name=lang.name,
                sentence_type_count=len(lang.sentence_types),
                sentence_types=[st.__name__ for st in lang.sentence_types],
            )
            for lang in languages
        ]
    )


@router.get("/{language_code}", response_model=LanguageDetail)
async def get_language_info(language_code: str):
    """Get detailed information about a specific language."""
    lang = get_language(language_code)
    return LanguageDetail(
        code=lang.code,
        name=lang.name,
        sentence_types=[
            SentenceTypeInfo(
                name=st.__name__,
                field_count=len(st.model_fields),
            )
            for st in lang.sentence_types
        ],
    )
