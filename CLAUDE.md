# yaduha-2

Type-safe, AI-powered framework for structured language translation using Pydantic models as grammar constraints with LLM structured output.

## Development Workflow

**Always run these checks before considering work complete:**

```bash
# Run all tests
uv run pytest tests/ -v

# Lint (must pass with 0 errors)
uv run ruff check yaduha/ tests/
uv run ruff format --check yaduha/ tests/

# Type checking (must pass with 0 errors)
uv run pyright yaduha/
uv run mypy yaduha/
```

To auto-fix import sorting and lint issues:

```bash
uv run ruff check yaduha/ tests/ --fix
uv run ruff format yaduha/ tests/
```

## Architecture

### Core Modules

- `yaduha/translator/` — Translation pipeline
  - `__init__.py` — `Translation`, `BackTranslation`, `Translator` base classes
  - `pipeline.py` — `PipelineTranslator` (grammar-guaranteed via structured output)
  - `instructions.py` — `InstructionsTranslator` (LLM with instructions, supports `back_translator` and `evaluators`)
  - `instructions_back.py` — `InstructionsBackTranslator` (back-translate using language source code as LLM instructions)
  - `agentic.py` — `AgenticTranslator` (free-form with tool assistance)
  - `back_translator.py` — `BackTranslator` ABC
- `yaduha/evaluator/` — Translation quality evaluation
  - `__init__.py` — `Evaluator` ABC, `OpenAIEvaluator`, `batch_evaluate()`
  - `chrf.py` — `ChrfEvaluator` (sacrebleu, `yaduha[eval]`)
  - `bleu.py` — `BleuEvaluator` (sacrebleu, `yaduha[eval]`)
  - `bertscore.py` — `BertScoreEvaluator` (bert-score, `yaduha[eval]`)
  - `comet.py` — `CometEvaluator` (unbabel-comet, `yaduha[eval]`)
- `yaduha/agent/` — LLM agent backends (OpenAI, Anthropic, etc.)
- `yaduha/language/` — Language class and loader
- `yaduha/tool/` — Tool base class for agents
- `yaduha/logger/` — Logging (JSON, W&B, print, no-op)
- `yaduha/api/` — FastAPI REST API

### Key Patterns

- All framework classes extend `pydantic.BaseModel`
- `Translation` has `evaluations: dict[str, float]` for self-contained evaluation scores
- Evaluators use lazy imports for heavy optional deps (`bert_score`, `comet`)
- `batch_evaluate(translations, evaluator)` returns new Translation copies with scores added
- `Sequence[Evaluator]` (not `List[Evaluator]`) for covariant type compatibility
- Language packages register via `[project.entry-points."yaduha.languages"]`

### Dependencies

- Core: `pydantic`, `openai`
- Eval metrics: `pip install yaduha[eval]` — sacrebleu, bert-score, unbabel-comet (heavy, NOT in `all` group)
- Dev: `pip install yaduha[dev]` — pytest, mypy, ruff, pyright

## Testing Conventions

- Tests live in `tests/` directory
- Mock heavy/optional deps (`bert_score`, `comet`) using `patch.dict(sys.modules, ...)`
- Mock Pydantic `Agent` fields using concrete subclasses (MagicMock fails Pydantic validation)
- Use `MagicMock(spec=[])` when you need `hasattr()` to return False on a mock

## Known Issues

- `test_cli.py::test_main_languages_search` — pre-existing failure (tests a `search` subcommand that doesn't exist in CLI)
- `eval` deps are NOT included in the `all` optional group because they pull in torch/CUDA (very heavy)

## Lint/Type Config

- **ruff**: `select = ["E", "F", "I", "UP"]`, `ignore = ["E501"]` (line length handled by ruff format)
- **pyright**: `typeCheckingMode = "basic"`, `reportMissingImports = "warning"`
- **mypy**: `ignore_missing_imports = true`, overrides ignore errors in `yaduha.agent.*`, `yaduha.api.*`, `yaduha.tool.*`
