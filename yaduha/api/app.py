"""FastAPI application factory."""

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from yaduha.api.routes import experiments, health, languages, schemas, translate


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Yaduha Translation API",
        description=(
            "A generic REST API for the Yaduha structured language translation framework. "
            "Works with any installed language package."
        ),
        version="0.3.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    api.include_router(languages.router)
    api.include_router(schemas.router)
    api.include_router(translate.router)
    api.include_router(experiments.router)
    app.include_router(api)

    return app
