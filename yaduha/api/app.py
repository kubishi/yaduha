"""FastAPI application factory."""

from fastapi import FastAPI

from yaduha.api.routes import health, languages, schemas, translate


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

    app.include_router(health.router)
    app.include_router(languages.router)
    app.include_router(schemas.router)
    app.include_router(translate.router)

    return app
