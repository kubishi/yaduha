"""MCP Server implementation using official Anthropic MCP SDK.

Provides proper JSON-RPC stdio transport for language development tools.
"""

import logging

from mcp.server import FastMCP

from yaduha.mcp_server.server import LanguageDevelopmentServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YaduhaLanguageDevelopmentServer:
    """MCP server wrapper for Yaduha language development tools."""

    def __init__(self, language_path: str = "yaduha-ovp"):
        """Initialize the MCP server."""
        self.mcp = FastMCP("yaduha-dev-server")
        self.language_server = LanguageDevelopmentServer(language_path)
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all language development tools with the MCP server."""

        @self.mcp.tool()
        def read_file(path: str) -> str:
            """Read a file from the language package."""
            return self.language_server.read_file(path)

        @self.mcp.tool()
        def write_file(path: str, content: str) -> str:
            """Write a file to the language package (auto-runs tests)."""
            return self.language_server.write_file(path, content)

        @self.mcp.tool()
        def delete_file(path: str) -> str:
            """Delete a file from the language package."""
            return self.language_server.delete_file(path)

        @self.mcp.tool()
        def list_package_structure() -> str:
            """List the structure of the language package."""
            return self.language_server.list_package_structure()

        @self.mcp.tool()
        def show_git_status() -> str:
            """Show git status of the language package."""
            return self.language_server.show_git_status()

        @self.mcp.tool()
        def run_tests() -> str:
            """Run pytest on the language package."""
            return self.language_server.run_tests()

        @self.mcp.tool()
        def translate_text(text: str) -> str:
            """Translate text using the language package."""
            return self.language_server.translate_text(text)

        @self.mcp.tool()
        def validate_language() -> str:
            """Validate the language package structure and syntax."""
            return self.language_server.validate_language()

        @self.mcp.tool()
        def commit_changes(message: str) -> str:
            """Commit changes to the language package git repository."""
            return self.language_server.commit_changes(message)

        @self.mcp.tool()
        def revert_file(path: str) -> str:
            """Revert a file to its last committed state."""
            return self.language_server.revert_file(path)

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        logger.info(f"✓ MCP Server initialized for {self.language_server.language_code}")
        await self.mcp.run_stdio_async()


def main() -> None:
    """Entry point for the MCP server."""
    import asyncio

    server = YaduhaLanguageDevelopmentServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
