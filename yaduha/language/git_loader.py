"""
Git-based Language Loader

Loads language modules from git repositories by:
1. Cloning or pulling the repo to ~/.yaduha/langs/[name]
2. Validating the language structure
3. Loading the language module
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from yaduha.language.loader import (
    load_language_from_path,
    validate_language_path,
    LoadedLanguage,
    LanguageLoadError,
)


# Default directory for cached language repos
DEFAULT_CACHE_DIR = Path.home() / ".yaduha" / "langs"


class GitLoadError(Exception):
    """Raised when a git operation fails."""
    pass


def get_repo_name(repo_url: str) -> str:
    """
    Extract a clean repo name from a git URL.

    Examples:
        https://github.com/user/my-lang.git -> my-lang
        git@github.com:user/my-lang.git -> my-lang
        https://github.com/user/my-lang -> my-lang
    """
    # Handle SSH URLs (git@github.com:user/repo.git)
    if repo_url.startswith("git@"):
        path = repo_url.split(":")[-1]
    else:
        parsed = urlparse(repo_url)
        path = parsed.path

    # Remove leading slash and .git suffix
    name = path.strip("/")
    if name.endswith(".git"):
        name = name[:-4]

    # Get just the repo name (last part of path)
    name = name.split("/")[-1]

    return name


def get_lang_path(repo_url: str, cache_dir: Optional[Path] = None) -> Path:
    """Get the local path where a repo would be cached."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    name = get_repo_name(repo_url)
    return cache_dir / name


def run_git_command(
    args: list[str],
    cwd: Optional[Path] = None,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a git command and handle errors.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess with the result

    Raises:
        GitLoadError: If the command fails
    """
    cmd = ["git"] + args

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else f"Command failed with code {result.returncode}"
            raise GitLoadError(f"Git command failed: {' '.join(cmd)}\n{error_msg}")

        return result

    except subprocess.TimeoutExpired:
        raise GitLoadError(f"Git command timed out: {' '.join(cmd)}")
    except FileNotFoundError:
        raise GitLoadError("Git is not installed or not in PATH")


def clone_repo(repo_url: str, dest_path: Path, branch: Optional[str] = None) -> None:
    """
    Clone a git repository.

    Args:
        repo_url: URL of the repository
        dest_path: Local path to clone to
        branch: Optional branch to checkout
    """
    # Ensure parent directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    args = ["clone", "--depth", "1"]  # Shallow clone for speed
    if branch:
        args.extend(["--branch", branch])
    args.extend([repo_url, str(dest_path)])

    run_git_command(args)


def pull_repo(repo_path: Path) -> bool:
    """
    Pull latest changes from remote.

    Args:
        repo_path: Local path to the repository

    Returns:
        True if there were updates, False if already up to date
    """
    # Fetch first
    run_git_command(["fetch"], cwd=repo_path)

    # Check if we're behind
    result = run_git_command(
        ["status", "-uno"],
        cwd=repo_path
    )

    if "Your branch is behind" in result.stdout:
        # Pull the changes
        run_git_command(["pull", "--ff-only"], cwd=repo_path)
        return True

    return False


def is_git_repo(path: Path) -> bool:
    """Check if a path is a git repository."""
    return (path / ".git").is_dir()


def load_language_from_git(
    repo_url: str,
    cache_dir: Optional[Path] = None,
    branch: Optional[str] = None,
    force_pull: bool = False,
    verbose: bool = True,
) -> Tuple[LoadedLanguage, Path]:
    """
    Load a language from a git repository.

    This will:
    1. Check if the repo already exists locally
    2. Clone or pull as needed
    3. Validate and load the language

    Args:
        repo_url: URL of the git repository
        cache_dir: Directory to cache repos (default: ~/.yaduha/langs)
        branch: Optional branch to checkout
        force_pull: Force pull even if repo exists
        verbose: Print status messages

    Returns:
        Tuple of (LoadedLanguage, local_path)

    Raises:
        GitLoadError: If git operations fail
        LanguageLoadError: If the language is invalid
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    repo_name = get_repo_name(repo_url)
    local_path = cache_dir / repo_name

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # Check if repo already exists
    if local_path.exists():
        if not is_git_repo(local_path):
            raise GitLoadError(
                f"Path exists but is not a git repo: {local_path}\n"
                f"Remove it manually if you want to re-clone."
            )

        log(f"Found existing repo at {local_path}")

        if force_pull:
            log("Pulling latest changes...")
            try:
                updated = pull_repo(local_path)
                if updated:
                    log("  Updated to latest version")
                else:
                    log("  Already up to date")
            except GitLoadError as e:
                log(f"  Warning: Pull failed: {e}")
                log("  Using existing local version")
    else:
        log(f"Cloning {repo_url}...")
        clone_repo(repo_url, local_path, branch)
        log(f"  Cloned to {local_path}")

    # Validate the language structure
    log("Validating language structure...")
    errors = validate_language_path(local_path)
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise LanguageLoadError(
            f"Language validation failed:\n{error_msg}"
        )
    log("  Structure is valid")

    # Load the language
    log("Loading language module...")
    lang = load_language_from_path(local_path)
    log(f"  Loaded '{lang.name}' ({lang.code})")
    log(f"  Sentence types: {[st.__name__ for st in lang.sentence_types]}")
    log(f"  Vocabulary: {len(lang.nouns)} nouns, {len(lang.transitive_verbs)} transitive verbs, {len(lang.intransitive_verbs)} intransitive verbs")

    return lang, local_path


def test_language(lang: LoadedLanguage, verbose: bool = True) -> bool:
    """
    Run basic tests on a loaded language.

    Args:
        lang: The loaded language to test
        verbose: Print status messages

    Returns:
        True if all tests pass, False otherwise
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    all_passed = True

    # Test 1: Examples exist and render
    log("Testing example rendering...")
    try:
        examples = lang.get_all_examples()
        if not examples:
            log("  FAILED: No examples found")
            all_passed = False
        else:
            for english, sentence in examples:
                rendered = str(sentence)
                if not rendered or not rendered.strip():
                    log(f"  FAILED: Empty render for '{english}'")
                    all_passed = False
                    break
            else:
                log(f"  PASSED: {len(examples)} examples render correctly")
                if verbose:
                    for english, sentence in examples[:3]:
                        log(f"    '{english}' -> '{str(sentence)}'")
                    if len(examples) > 3:
                        log(f"    ... and {len(examples) - 3} more")
    except Exception as e:
        log(f"  FAILED: {e}")
        all_passed = False

    # Test 2: JSON serialization
    log("Testing JSON serialization...")
    try:
        for st in lang.sentence_types:
            examples = st.get_examples()
            for _, sentence in examples[:2]:
                json_data = sentence.model_dump()
                reconstructed = st.model_validate(json_data)
                if str(reconstructed) != str(sentence):
                    log(f"  FAILED: Serialization mismatch for {st.__name__}")
                    all_passed = False
                    break
            else:
                continue
            break
        else:
            log("  PASSED: All sentence types serialize correctly")
    except Exception as e:
        log(f"  FAILED: {e}")
        all_passed = False

    # Test 3: Vocabulary integrity
    log("Testing vocabulary...")
    try:
        # Check for duplicates
        all_vocab = (
            lang.nouns +
            lang.transitive_verbs +
            lang.intransitive_verbs +
            lang.adjectives +
            lang.adverbs
        )

        english_words = [v.english for v in all_vocab]
        duplicates = [w for w in set(english_words) if english_words.count(w) > 1]

        if duplicates:
            log(f"  WARNING: Duplicate English words: {duplicates[:5]}")

        # Check for empty entries
        empty = [v for v in all_vocab if not v.english or not v.target]
        if empty:
            log(f"  FAILED: {len(empty)} empty vocabulary entries")
            all_passed = False
        else:
            log(f"  PASSED: {len(all_vocab)} vocabulary entries are valid")
    except Exception as e:
        log(f"  FAILED: {e}")
        all_passed = False

    return all_passed


def remove_cached_language(
    repo_url_or_name: str,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> bool:
    """
    Remove a cached language repository.

    Args:
        repo_url_or_name: Either a repo URL or just the repo name
        cache_dir: Directory where repos are cached
        verbose: Print status messages

    Returns:
        True if removed, False if not found
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Determine the path
    if "/" in repo_url_or_name or repo_url_or_name.startswith("git@"):
        name = get_repo_name(repo_url_or_name)
    else:
        name = repo_url_or_name

    local_path = cache_dir / name

    if not local_path.exists():
        if verbose:
            print(f"Language not found: {local_path}")
        return False

    if verbose:
        print(f"Removing {local_path}...")

    shutil.rmtree(local_path)

    if verbose:
        print("  Removed")

    return True


def list_cached_languages(
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> list[Path]:
    """
    List all cached language repositories.

    Args:
        cache_dir: Directory where repos are cached
        verbose: Print status messages

    Returns:
        List of paths to cached languages
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    if not cache_dir.exists():
        if verbose:
            print(f"No cached languages (cache dir doesn't exist: {cache_dir})")
        return []

    languages = []
    for path in cache_dir.iterdir():
        if path.is_dir() and is_git_repo(path):
            languages.append(path)

    if verbose:
        if languages:
            print(f"Cached languages in {cache_dir}:")
            for path in languages:
                print(f"  - {path.name}")
        else:
            print(f"No cached languages in {cache_dir}")

    return languages
