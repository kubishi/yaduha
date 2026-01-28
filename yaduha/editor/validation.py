"""
Validation Feedback

When tests fail or validation errors occur, uses LLM to explain what's wrong
and suggest fixes in plain language.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path
import traceback

from yaduha.agent import Agent
from yaduha.language import LoadedLanguage, LanguageLoadError

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class IssueSeverity(str, Enum):
    """Severity level of a validation issue."""
    error = "error"  # Prevents the language from working
    warning = "warning"  # May cause problems but not fatal
    suggestion = "suggestion"  # Improvement opportunity


class IssueCategory(str, Enum):
    """Category of validation issue."""
    missing_export = "missing_export"
    invalid_type = "invalid_type"
    missing_method = "missing_method"
    example_error = "example_error"
    rendering_error = "rendering_error"
    vocabulary_error = "vocabulary_error"
    syntax_error = "syntax_error"
    import_error = "import_error"
    logic_error = "logic_error"
    style_issue = "style_issue"


class ValidationIssue(BaseModel):
    """A single validation issue found in the language."""
    severity: IssueSeverity = Field(..., description="How severe the issue is")
    category: IssueCategory = Field(..., description="Category of the issue")
    location: str = Field(..., description="Where the issue was found (file:line or component name)")
    message: str = Field(..., description="Technical description of the issue")
    explanation: str = Field(..., description="Plain language explanation of what's wrong")
    context: Optional[str] = Field(default=None, description="Relevant code or context")


class ValidationFix(BaseModel):
    """A suggested fix for a validation issue."""
    issue_summary: str = Field(..., description="Brief summary of what's being fixed")
    explanation: str = Field(..., description="Plain language explanation of the fix")
    code_before: Optional[str] = Field(default=None, description="Code that needs to change")
    code_after: Optional[str] = Field(default=None, description="Corrected code")
    file_path: Optional[str] = Field(default=None, description="File to modify")
    steps: List[str] = Field(default_factory=list, description="Step-by-step instructions")
    related_docs: List[str] = Field(default_factory=list, description="Links to relevant documentation")


class ValidationReport(BaseModel):
    """Complete validation report with issues and fixes."""
    is_valid: bool = Field(..., description="Whether the language passes validation")
    issues: List[ValidationIssue] = Field(default_factory=list, description="All issues found")
    fixes: List[ValidationFix] = Field(default_factory=list, description="Suggested fixes")
    summary: str = Field(..., description="High-level summary of validation results")


class TestFailureAnalysis(BaseModel):
    """Analysis of a test failure."""
    test_name: str = Field(..., description="Name of the failing test")
    error_type: str = Field(..., description="Type of error that occurred")
    root_cause: str = Field(..., description="Identified root cause")
    explanation: str = Field(..., description="Plain language explanation")
    suggested_fix: ValidationFix = Field(..., description="How to fix it")


class _FixList(BaseModel):
    """Internal model for fix list response."""
    fixes: List[ValidationFix]


class _Checklist(BaseModel):
    """Internal model for checklist response."""
    items: List[str]


class ValidationFeedback:
    """
    LLM-powered feedback for validation errors and test failures.

    Translates technical errors into actionable, understandable feedback
    for users creating their own languages.
    """

    def __init__(self, agent: Agent[Any], language: Optional[LoadedLanguage] = None):
        """
        Initialize the validation feedback system.

        Args:
            agent: The LLM agent to use for generating feedback
            language: Optional loaded language for context
        """
        self.agent = agent
        self.language = language

    def _get_language_context(self) -> str:
        """Get context about the language structure."""
        if not self.language:
            return "No language context available."

        lines = [f"# {self.language.name} ({self.language.code})"]

        lines.append(f"\nSentence types: {len(self.language.sentence_types)}")
        for st in self.language.sentence_types:
            lines.append(f"- {st.__name__}")

        lines.append("\nVocabulary counts:")
        lines.append(f"- Nouns: {len(self.language.nouns)}")
        lines.append(f"- Transitive verbs: {len(self.language.transitive_verbs)}")
        lines.append(f"- Intransitive verbs: {len(self.language.intransitive_verbs)}")

        return "\n".join(lines)

    def explain_load_error(
        self,
        error: LanguageLoadError,
        language_path: Optional[Path] = None,
    ) -> ValidationReport:
        """
        Explain a language loading error and suggest fixes.

        Args:
            error: The LanguageLoadError that occurred
            language_path: Path to the language directory

        Returns:
            Validation report with explanation and fixes
        """
        system_prompt = """You are a helpful assistant explaining errors in constructed language definitions.
Your audience is language creators who may not be expert programmers.

Explain errors in plain language and provide actionable fixes with example code.
Be specific about what file to edit and what changes to make."""

        error_details = f"""Error loading language{f' from {language_path}' if language_path else ''}:

{str(error)}

Traceback:
{traceback.format_exc() if hasattr(error, '__traceback__') else 'Not available'}"""

        user_content = f"""A user got this error when loading their language:

{error_details}

Please:
1. Explain what went wrong in plain language
2. Identify the specific issues
3. Provide step-by-step fix instructions with code examples"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=ValidationReport
        )

        return response.content

    def explain_validation_errors(
        self,
        errors: List[str],
        source_code: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> ValidationReport:
        """
        Explain validation errors and suggest fixes.

        Args:
            errors: List of error messages
            source_code: Optional relevant source code
            file_path: Optional path to the file with errors

        Returns:
            Validation report with explanations and fixes
        """
        system_prompt = """You are a helpful assistant explaining validation errors in constructed language definitions.
Your audience is language creators who may not be expert programmers.

For each error:
1. Explain what the error means in plain language
2. Explain why it matters
3. Show exactly how to fix it with code examples
4. Mention any related issues to watch for"""

        context = self._get_language_context()

        user_content = f"""A user's language has these validation errors:

{chr(10).join(f'- {e}' for e in errors)}

{f'File: {file_path}' if file_path else ''}

{f'Source code:{chr(10)}{source_code}' if source_code else ''}

Language context:
{context}

Explain each error and how to fix it."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=ValidationReport
        )

        return response.content

    def explain_test_failure(
        self,
        test_name: str,
        error_message: str,
        test_code: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ) -> TestFailureAnalysis:
        """
        Explain why a test failed and how to fix it.

        Args:
            test_name: Name of the failing test
            error_message: The error message
            test_code: Optional test code
            traceback_str: Optional traceback

        Returns:
            Analysis of the failure with fix
        """
        system_prompt = """You are a helpful assistant explaining test failures in constructed language implementations.
Your audience is language creators who may not be expert programmers.

Analyze the test failure and explain:
1. What the test was checking
2. Why it failed (root cause)
3. What code needs to change
4. Step-by-step fix instructions"""

        context = self._get_language_context()

        user_content = f"""A test failed:

Test: {test_name}
Error: {error_message}

{f'Traceback:{chr(10)}{traceback_str}' if traceback_str else ''}

{f'Test code:{chr(10)}{test_code}' if test_code else ''}

Language context:
{context}

Analyze this failure and explain how to fix it."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=TestFailureAnalysis
        )

        return response.content

    def explain_rendering_error(
        self,
        sentence_type: str,
        input_data: Dict[str, Any],
        error: Exception,
        expected_output: Optional[str] = None,
    ) -> ValidationFix:
        """
        Explain why a sentence failed to render and suggest fixes.

        Args:
            sentence_type: Name of the sentence type
            input_data: Input that caused the error
            error: The exception that occurred
            expected_output: What the output should have been

        Returns:
            Fix suggestion
        """
        system_prompt = """You are a helpful assistant debugging sentence rendering in constructed languages.
Your audience is language creators who may not be expert programmers.

Explain:
1. What the rendering code was trying to do
2. Why it failed
3. How to fix the __str__ method or related code"""

        context = self._get_language_context()

        user_content = f"""Sentence rendering failed:

Sentence type: {sentence_type}
Input: {input_data}
Error: {type(error).__name__}: {str(error)}
{f'Expected output: {expected_output}' if expected_output else ''}

Language context:
{context}

Explain the problem and how to fix it."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=ValidationFix
        )

        return response.content

    def review_code(
        self,
        code: str,
        file_type: str = "language definition",
    ) -> ValidationReport:
        """
        Review language code for potential issues before running.

        Args:
            code: The code to review
            file_type: Type of file being reviewed

        Returns:
            Validation report with any issues found
        """
        system_prompt = """You are a code reviewer for constructed language implementations.
Review the code for:
1. Missing required exports (LANGUAGE_NAME, LANGUAGE_CODE, SENTENCE_TYPES)
2. Sentence types that don't properly implement __str__() or get_examples()
3. Type annotation issues
4. Pydantic model problems
5. Vocabulary consistency
6. Common bugs and edge cases

Be thorough but focus on issues that will cause actual problems."""

        context = self._get_language_context()

        user_content = f"""Review this {file_type}:

```python
{code}
```

Language context:
{context}

Identify any issues and suggest improvements."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=ValidationReport
        )

        return response.content

    def suggest_missing_components(
        self,
        current_exports: List[str],
    ) -> List[ValidationFix]:
        """
        Suggest what's missing from a language definition.

        Args:
            current_exports: List of currently exported names

        Returns:
            List of fixes for missing components
        """
        required = ["LANGUAGE_NAME", "LANGUAGE_CODE", "SENTENCE_TYPES"]
        missing = [r for r in required if r not in current_exports]

        if not missing:
            return []

        system_prompt = """You are a helpful assistant for constructed language creation.
Generate example code for missing required components."""

        user_content = f"""A language definition is missing these required exports:
{chr(10).join(f'- {m}' for m in missing)}

Current exports: {', '.join(current_exports)}

Provide example code for each missing component."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=_FixList
        )

        return response.content.fixes

    def generate_validation_checklist(self) -> List[str]:
        """
        Generate a validation checklist for language creators.

        Returns:
            List of items to check before publishing a language
        """
        system_prompt = """You are creating a validation checklist for constructed language implementations.
Include checks for:
1. Required exports and structure
2. Sentence type implementation
3. Vocabulary coverage
4. Example quality
5. Common pitfalls"""

        context = self._get_language_context()

        user_content = f"""Generate a validation checklist for this language:

{context}

Create a comprehensive but practical checklist."""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self.agent.get_response(
            messages=messages,
            response_format=_Checklist
        )

        return response.content.items
