"""FastAPI application factory."""

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

    # Mount all API routes under /api
    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    api.include_router(languages.router)
    api.include_router(schemas.router)
    api.include_router(translate.router)
    app.include_router(api)

    # Serve built frontend if dist/ exists
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if frontend_dir.is_dir():
        assets_dir = frontend_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        index_html = frontend_dir / "index.html"

        @app.get("/{path:path}")
        async def serve_spa(path: str):
            # Serve actual files if they exist in dist/
            file_path = frontend_dir / path
            if path and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(index_html))

    return app
