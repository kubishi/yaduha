"""
Test script for dynamically loaded conlangs.

This script:
1. Loads a language from a local path or git repository
2. Runs validation tests on the language
3. Optionally runs a live translation test (requires ANTHROPIC_API_KEY)

Usage:
    # Load from local path
    python scripts/test_conlang.py ./conlangs/ovp-lang

    # Load from git repo
    python scripts/test_conlang.py https://github.com/user/my-conlang

    # Force pull latest changes
    python scripts/test_conlang.py https://github.com/user/my-conlang --pull

    # Run live translation tests
    python scripts/test_conlang.py ./conlangs/ovp-lang --live

    # List cached languages
    python scripts/test_conlang.py --list

    # Remove a cached language
    python scripts/test_conlang.py --remove my-conlang
"""

import argparse
import os
import sys
import dotenv
import pathlib

from yaduha.language import (
    load_language_from_path,
    load_language_from_git,
    validate_language_path,
    test_language,
    list_cached_languages,
    remove_cached_language,
    LanguageLoadError,
    GitLoadError,
    LoadedLanguage,
)
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.anthropic import AnthropicAgent

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()
DEFAULT_LANG_URL = "https://github.com/kubishi/ovp-lang"


def is_git_url(path: str) -> bool:
    """Check if a string looks like a git URL."""
    return (
        path.startswith("https://") or
        path.startswith("http://") or
        path.startswith("git@") or
        path.startswith("git://")
    )


def load_language(
    source: str,
    force_pull: bool = False,
) -> tuple[LoadedLanguage, pathlib.Path]:
    """
    Load a language from either a local path or git URL.

    Returns:
        Tuple of (LoadedLanguage, path)
    """
    if is_git_url(source):
        print(f"Loading language from git: {source}")
        print()
        lang, path = load_language_from_git(
            source,
            force_pull=force_pull,
            verbose=True,
        )
        return lang, path
    else:
        path = pathlib.Path(source).resolve()
        print(f"Loading language from path: {path}")
        print()

        # Validate first
        print("Validating language structure...")
        errors = validate_language_path(path)
        if errors:
            for e in errors:
                print(f"  - {e}")
            raise LanguageLoadError(f"Validation failed with {len(errors)} errors")
        print("  Structure is valid")

        # Load
        print("Loading language module...")
        lang = load_language_from_path(path)
        print(f"  Loaded '{lang.name}' ({lang.code})")
        print(f"  Sentence types: {[st.__name__ for st in lang.sentence_types]}")
        print(f"  Vocabulary: {len(lang.nouns)} nouns, {len(lang.transitive_verbs)} transitive verbs, {len(lang.intransitive_verbs)} intransitive verbs")

        return lang, path


def test_translation(lang: LoadedLanguage, verbose: bool = True) -> bool:
    """Test live translation (requires ANTHROPIC_API_KEY)."""
    print("Testing: Live translation")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIPPED: ANTHROPIC_API_KEY not set")
        return True

    try:
        agent = AnthropicAgent(
            api_key=api_key,
            model="claude-sonnet-4-20250514"
        )
        translator = PipelineTranslator(
            agent=agent,
            SentenceType=lang.sentence_types,
        )

        test_sentences = [
            "I sleep.",
            "The coyote runs.",
            "The dog eats the fish.",
        ]

        for sentence in test_sentences:
            result = translator.translate(sentence)
            print(f"    '{sentence}'")
            print(f"      -> '{result.target}'")
            if result.back_translation:
                print(f"      <- '{result.back_translation.source}'")

        print("  PASSED: Translation completed")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def run_tests(
    lang: LoadedLanguage,
    include_live: bool = False,
) -> bool:
    """Run all tests on the loaded language."""
    print()
    print("=" * 60)
    print("Running language tests")
    print("=" * 60)
    print()

    results = []

    # Run standard tests
    passed = test_language(lang, verbose=True)
    results.append(("language_tests", passed))

    print()

    # Run translation test if requested
    if include_live:
        passed = test_translation(lang)
        results.append(("live_translation", passed))
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(r for _, r in results)
    passed_count = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print()
    print(f"Total: {passed_count}/{total} test groups passed")

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load and test a conlang from a local path or git repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./conlangs/ovp-lang          Load from local path
  %(prog)s https://github.com/u/repo    Load from git repo
  %(prog)s https://github.com/u/repo -p Force pull latest
  %(prog)s ./conlangs/ovp-lang --live   Include translation tests
  %(prog)s --list                       List cached languages
  %(prog)s --remove my-lang             Remove cached language
        """
    )

    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_LANG_URL,
        help="Path to language directory or git URL (default: conlangs/ovp-lang)"
    )
    parser.add_argument(
        "-p", "--pull",
        action="store_true",
        help="Force pull latest changes (for git repos)"
    )
    parser.add_argument(
        "-l", "--live",
        action="store_true",
        help="Include live translation tests (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached languages and exit"
    )
    parser.add_argument(
        "--remove",
        metavar="NAME",
        help="Remove a cached language by name or URL"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_cached_languages(verbose=True)
        return 0

    # Handle --remove
    if args.remove:
        if remove_cached_language(args.remove, verbose=True):
            return 0
        else:
            return 1

    # Load and test the language
    try:
        print("=" * 60)
        print("Loading conlang")
        print("=" * 60)
        print()

        lang, _ = load_language(args.source, force_pull=args.pull)

        success = run_tests(lang, include_live=args.live)

        return 0 if success else 1

    except LanguageLoadError as e:
        print()
        print(f"ERROR: Language load failed")
        print(f"  {e}")
        return 1

    except GitLoadError as e:
        print()
        print(f"ERROR: Git operation failed")
        print(f"  {e}")
        return 1

    except KeyboardInterrupt:
        print()
        print("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
