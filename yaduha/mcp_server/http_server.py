"""HTTP wrapper for Language Development MCP Server.

Provides FastAPI-based HTTP interface to the MCP server tools.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from yaduha.mcp_server.server import LanguageDevelopmentServer


app = FastAPI(title="Yaduha MCP Server")
server: Optional[LanguageDevelopmentServer] = None


class FileContent(BaseModel):
    """Request body for file operations."""
    content: str


@app.on_event("startup")
async def startup(language_path: str = "yaduha-ovp"):
    """Initialize the server on startup."""
    global server
    server = LanguageDevelopmentServer(language_path)
    print(f"✓ Server initialized for {server.language_code}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return {
        "status": "healthy",
        "language": server.language_code,
        "name": server.language_name,
        "path": str(server.language_path),
    }


@app.get("/files/{path:path}")
async def read_file(path: str):
    """Read a file from the language package."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.read_file(path)
    if "Error" in result and path not in ["__init__.py", "vocab.py"]:
        raise HTTPException(status_code=400, detail=result)
    return {"content": result}


@app.post("/files/{path:path}")
async def write_file(path: str, body: FileContent):
    """Write a file to the language package."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.write_file(path, body.content)
    if "Error" in result:
        raise HTTPException(status_code=400, detail=result)
    return {"message": result}


@app.delete("/files/{path:path}")
async def delete_file(path: str):
    """Delete a file from the language package."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.delete_file(path)
    if "Error" in result:
        raise HTTPException(status_code=400, detail=result)
    return {"message": result}


@app.get("/structure")
async def list_structure():
    """Get package structure."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.list_package_structure()
    files = result.split("\n") if result else []
    return {"files": files}


@app.get("/git/status")
async def git_status():
    """Get git status."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.show_git_status()
    return {"status": result}


@app.post("/git/commit")
async def commit(body: BaseModel):
    """Commit changes."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Handle both dict and model
    message = body.message if hasattr(body, "message") else body.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Message required")

    result = server.commit_changes(message)
    if "Error" in result:
        raise HTTPException(status_code=400, detail=result)
    return {"message": result}


@app.post("/git/revert/{path:path}")
async def revert_file(path: str):
    """Revert a file."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.revert_file(path)
    if "Error" in result:
        raise HTTPException(status_code=400, detail=result)
    return {"message": result}


@app.get("/tests")
async def run_tests():
    """Run tests."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.run_tests()
    return {"output": result}


@app.get("/translate")
async def translate_text(text: str):
    """Translate text."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.translate_text(text)
    return {"translation": result}


@app.get("/validate")
async def validate():
    """Validate language."""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    result = server.validate_language()
    return {"validation": result}


@app.get("/tools")
async def list_tools():
    """List available tools."""
    return {
        "tools": [
            {"name": "read_file", "description": "Read a file from the language package"},
            {"name": "write_file", "description": "Write a file (auto-runs tests)"},
            {"name": "delete_file", "description": "Delete a file"},
            {"name": "list_package_structure", "description": "List package files"},
            {"name": "git_status", "description": "Show git status"},
            {"name": "commit", "description": "Commit changes"},
            {"name": "revert_file", "description": "Revert a file"},
            {"name": "run_tests", "description": "Run tests"},
            {"name": "translate_text", "description": "Translate text"},
            {"name": "validate", "description": "Validate language"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
