"""MCP Server for Language Development.

Exposes language development tools to LLMs via Model Context Protocol.
"""

from yaduha.mcp_server.server import LanguageDevelopmentServer

__all__ = ["LanguageDevelopmentServer"]
