# Phase 1 Implementation: Language Package Format & Installation

## Context & Goals

This phase implements the core infrastructure for Yaduha language packages, allowing external language communities to:
1. Define their own sentence types (linguistic structures)
2. Package them as Python packages with entrypoints
3. Install and use them with Yaduha's translation tools
4. Validate their language implementations

### Key Design Decisions

- **Entrypoints**: Languages register via `[project.entry-points."yaduha.languages"]` in `pyproject.toml` (Python standard)
- **Language Class**: Minimal wrapper for sentence types, enables runtime discovery
- **LanguageLoader**: Central registry lookup via entrypoints (no naming conventions)
- **Code Quality**: All code must be fully typed, pass pylint/mypy, 90%+ test coverage

## Files Created/Modified

### Core Implementation
- `yaduha/language/language.py` - Language class (wraps sentence types)
- `yaduha/language/exceptions.py` - Custom exceptions (LanguageNotFoundError, LanguageValidationError)
- `yaduha/loader.py` - LanguageLoader (discover/load/validate languages)
- `yaduha/cli.py` - Command-line interface
- `yaduha/language/__init__.py` - Updated to export Language, exceptions
- `yaduha/__init__.py` - Updated to export LanguageLoader
- `pyproject.toml` - Added `[project.scripts]` entry for `yaduha` command

### Tests (32 total, all passing)
- `tests/test_language.py` - 8 Language class tests
- `tests/test_loader.py` - 13 LanguageLoader tests
- `tests/test_pipeline_translator_from_language.py` - 2 PipelineTranslator tests
- `tests/test_cli.py` - 9 CLI tests

### Modified Existing Code
- `yaduha/translator/pipeline.py` - Added `from_language()` classmethod for convenience

## Next Steps (Phase 1 Remaining)

### Step 2: CLI Commands ✅ DONE
- ✅ `yaduha languages list` - Show installed languages
- ✅ `yaduha languages info CODE` - Show language details
- ✅ `yaduha languages validate CODE` - Validate language implementation
- ✅ `yaduha languages search` - Search registry placeholder

### Step 3: Move OVP to Separate Repository ✅ DONE
- ✅ Created `kubishi/yaduha-ovp` repository (https://github.com/kubishi/yaduha-ovp)
- ✅ Migrated `yaduha/language/ovp/` code to separate package with proper structure
- ✅ Added entrypoint in `yaduha-ovp/pyproject.toml`: `ovp = "yaduha_ovp:language"`
- ✅ Removed OVP from main yaduha repository
- ✅ Verified OVP is discoverable via `yaduha languages list` and validation works

### Step 4: Integration Testing (IN PROGRESS)
- Test OVP as installed package ✅ (verified with `yaduha languages list`, info, validate)
- End-to-end translation tests ⏳
- Verify `yaduha languages list` discovers OVP ✅

### Step 5: Documentation & Registry (TODO)
- Write "Creating a Language Package" guide
- Create language registry YAML
- Update existing docs

## Running Tests

```bash
# Using uv (recommended) - run all Phase 1 tests
uv run pytest tests/test_language.py tests/test_loader.py tests/test_pipeline_translator_from_language.py tests/test_cli.py -v

# Run with coverage
uv run pytest tests/test_language.py tests/test_loader.py tests/test_pipeline_translator_from_language.py tests/test_cli.py --cov=yaduha/language --cov=yaduha/loader --cov=yaduha/cli

# Run specific test module
uv run pytest tests/test_cli.py -v
```

## CLI Usage

```bash
# List installed languages
uv run yaduha languages list

# Show language details
uv run yaduha languages info ovp

# Validate a language
uv run yaduha languages validate ovp

# Show help
uv run yaduha languages --help
uv run yaduha languages info --help
```

## Implementation Principles

1. **Type Safety**: Every parameter and return type must be annotated (no `Any`)
2. **Simplicity**: Favor readable code over cleverness; keep functions small
3. **Testing**: Every public method has tests with mocked dependencies
4. **Documentation**: Docstrings for all classes/functions, comments for non-obvious logic
5. **Quality Gates**: Must pass pylint, mypy --strict, pytest with 90%+ coverage

## Design Notes

- **Language as wrapper**: Language class is intentionally minimal - just code, name, sentence_types. Metadata can live in packages' pyproject.toml.
- **Entrypoints**: Using standard Python entrypoints enables flexibility (languages from any package, not just `yaduha-*` naming)
- **LanguageLoader validation**: Comprehensive but focused on structural correctness (has __str__, get_examples, valid Pydantic models)
- **PipelineTranslator.from_language()**: Convenience method that loads language and creates translator in one call

## Step 3 Details: yaduha-ovp Repository

The yaduha-ovp repository demonstrates the pattern for creating external language packages:

### Repository Structure
```
kubishi/yaduha-ovp/
├── pyproject.toml                    # Package config with entrypoint
├── README.md                         # Package documentation
├── LICENSE.md                        # License file
├── .gitignore                        # Git ignore patterns
├── yaduha_ovp/
│   ├── __init__.py                   # Main module exporting Language instance
│   ├── vocab.py                      # Vocabulary data (NOUNS, VERBS)
│   └── prompts.py                    # Prompt generation utilities
```

### Key Configuration
- **Entrypoint**: `ovp = "yaduha_ovp:language"` in `[project.entry-points."yaduha.languages"]`
- **Module Export**: `yaduha_ovp/__init__.py` exports a Language instance: `language = Language(code="ovp", ...)`
- **Dependencies**: Depends on yaduha>=0.3 and pydantic

### Testing the Package
```bash
# Install the package (editable mode for development)
pip install -e path/to/yaduha-ovp

# Verify discovery
yaduha languages list           # Should show "ovp"
yaduha languages info ovp       # Show OVP details
yaduha languages validate ovp   # Verify implementation
```
