"""Command-line interface for Yaduha language management."""

import argparse
import sys
from typing import Any, List

from yaduha.language import LanguageNotFoundError
from yaduha.loader import LanguageLoader


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



def cmd_serve(args: Any) -> int:
    """Start the Yaduha server (API + dashboard)."""
    import uvicorn
    from yaduha.api import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


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

    # serve command (starts both API and dashboard)
    serve_parser = subparsers.add_parser(
        "serve", help="Start the server (API + dashboard)"
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
