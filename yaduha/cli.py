"""Command-line interface for Yaduha language management."""

import argparse
import sys
from typing import Any, List

from yaduha.language import LanguageNotFoundError
from yaduha.loader import LanguageLoader
from yaduha.mcp_server import LanguageDevelopmentServer


def cmd_list(_args: Any) -> int:
    """List all installed languages."""
    languages = LanguageLoader.list_installed_languages()

    if not languages:
        print("No languages installed.")
        return 0

    print(f"\nFound {len(languages)} language(s):\n")
    for lang in languages:
        print(f"  {lang.code:10} - {lang.name:30} ({len(lang.sentence_types)} sentence types)")
    print()
    return 0


def cmd_info(args: Any) -> int:
    """Show language details."""
    try:
        lang = LanguageLoader.load_language(args.code)
    except LanguageNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nLanguage: {lang.name}")
    print(f"Code: {lang.code}")
    print(f"Sentence types: {len(lang.sentence_types)}")
    for i, sent_type in enumerate(lang.sentence_types, 1):
        print(f"  {i}. {sent_type.__name__}")
    print()
    return 0


def cmd_validate(args: Any) -> int:
    """Validate a language implementation."""
    is_valid, errors = LanguageLoader.validate_language(args.code)

    if is_valid:
        print(f"\n✓ Language '{args.code}' is valid!\n")
        return 0

    print(f"\n✗ Language '{args.code}' has errors:\n")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print()
    return 1


def cmd_search(args: Any) -> int:
    """Search the language registry (not yet implemented)."""
    print("\nLanguage registry search not yet implemented.\n")
    print("To install a language package:")
    print("  pip install yaduha-LANG_CODE")
    print("  or")
    print("  pip install git+https://github.com/user/yaduha-lang-code\n")
    return 0


def cmd_dev_server(args: Any) -> int:
    """Start the MCP language development server using stdio transport."""
    try:
        import asyncio
        from yaduha.mcp_server.http_server import YaduhaLanguageDevelopmentServer

        # Create the MCP server
        mcp_server = YaduhaLanguageDevelopmentServer(args.path)

        print(f"🚀 Starting Yaduha MCP Server")
        print(f"   Language: {mcp_server.language_server.language_code} - {mcp_server.language_server.language_name}")
        print(f"   Path: {mcp_server.language_server.language_path}")
        print(f"\n✓ Available tools: 10")
        print(f"✓ Server ready for MCP connections via stdio transport")
        print(f"\nConnect with: claude mcp add yaduha-dev-server ./your-script.py")
        print(f"Press Ctrl+C to stop the server.\n")

        # Run the MCP server
        asyncio.run(mcp_server.run())
        return 0
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main(argv: List[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="yaduha",
        description="Yaduha language management",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # languages subcommand group
    languages_parser = subparsers.add_parser(
        "languages",
        help="Manage languages",
    )
    languages_subparsers = languages_parser.add_subparsers(
        dest="subcommand",
        help="Languages subcommand",
    )

    # languages list
    list_parser = languages_subparsers.add_parser("list", help="List installed languages")
    list_parser.set_defaults(func=cmd_list)

    # languages info
    info_parser = languages_subparsers.add_parser("info", help="Show language info")
    info_parser.add_argument("code", help="Language code (e.g., 'ovp')")
    info_parser.set_defaults(func=cmd_info)

    # languages validate
    validate_parser = languages_subparsers.add_parser("validate", help="Validate language")
    validate_parser.add_argument("code", help="Language code (e.g., 'ovp')")
    validate_parser.set_defaults(func=cmd_validate)

    # languages search
    search_parser = languages_subparsers.add_parser("search", help="Search registry")
    search_parser.add_argument(
        "--query", "-q", help="Search query", default=None
    )
    search_parser.set_defaults(func=cmd_search)

    # dev-server command
    dev_server_parser = subparsers.add_parser(
        "dev-server",
        help="Start MCP language development server",
    )
    dev_server_parser.add_argument(
        "path",
        help="Path to language package directory",
    )
    dev_server_parser.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)",
    )
    dev_server_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port (default: 5000)",
    )
    dev_server_parser.set_defaults(func=cmd_dev_server)

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
