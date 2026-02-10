"""Health check route."""

from fastapi import APIRouter
from pydantic import BaseModel

from yaduha.loader import LanguageLoader

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    languages_available: int


@router.get("/health", response_model=HealthResponse)
async def health():
    langs = LanguageLoader.list_installed_languages()
    return HealthResponse(status="ok", languages_available=len(langs))
