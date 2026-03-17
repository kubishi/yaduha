"""Tests for yaduha.tool module: Tool base class validation, schema, and invocation."""

from typing import ClassVar

import pytest
from pydantic import BaseModel

from yaduha.logger import get_log_context
from yaduha.tool import Tool

# ---------------------------------------------------------------------------
# Concrete test tools
# ---------------------------------------------------------------------------


class EchoTool(Tool[str]):
    """Simple tool that echoes its input."""

    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Echo the input string."

    def _run(self, text: str) -> str:
        return f"echo: {text}"

    def get_examples(self) -> list[tuple[str, str]]:
        return [("hello", "echo: hello")]


class PersonInput(BaseModel):
    name: str
    age: int


class PersonTool(Tool[str]):
    """Tool that accepts a BaseModel arg."""

    name: ClassVar[str] = "greet_person"
    description: ClassVar[str] = "Greet a person."

    def _run(self, person: PersonInput) -> str:
        return f"Hello {person.name}, age {person.age}"

    def get_examples(self) -> list[tuple[PersonInput, str]]:
        return [(PersonInput(name="Alice", age=30), "Hello Alice, age 30")]


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


def test_tool_rejects_invalid_identifier():
    with pytest.raises(ValueError, match="not a valid Python identifier"):

        class BadTool(Tool[str]):
            name: ClassVar[str] = "bad-name"
            description: ClassVar[str] = "bad"

            def _run(self, x: str) -> str:
                return x

        BadTool()


def test_tool_rejects_builtin_name():
    with pytest.raises(ValueError, match="reserved"):

        class BuiltinTool(Tool[str]):
            name: ClassVar[str] = "print"
            description: ClassVar[str] = "bad"

            def _run(self, x: str) -> str:
                return x

        BuiltinTool()


# ---------------------------------------------------------------------------
# __call__ and _run
# ---------------------------------------------------------------------------


def test_tool_call_invokes_run():
    tool = EchoTool()
    result = tool("world")
    assert result == "echo: world"


def test_tool_call_with_kwargs():
    tool = EchoTool()
    result = tool(text="kwargs work")
    assert result == "echo: kwargs work"


def test_tool_auto_parses_basemodel_from_dict():
    tool = PersonTool()
    result = tool(person={"name": "Bob", "age": 25})
    assert result == "Hello Bob, age 25"


def test_tool_accepts_basemodel_directly():
    tool = PersonTool()
    result = tool(person=PersonInput(name="Carol", age=40))
    assert result == "Hello Carol, age 40"


# ---------------------------------------------------------------------------
# get_tool_call_schema
# ---------------------------------------------------------------------------


def test_tool_call_schema_structure():
    tool = EchoTool()
    schema = tool.get_tool_call_schema()

    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "echo"
    assert func["description"] == "Echo the input string."
    assert func["strict"] is True

    params = func["parameters"]
    assert "text" in params["properties"]
    assert params["additionalProperties"] is False
    assert "text" in params.get("required", [])


def test_tool_call_schema_basemodel_param():
    tool = PersonTool()
    schema = tool.get_tool_call_schema()
    params = schema["function"]["parameters"]
    assert "person" in params["properties"]


# ---------------------------------------------------------------------------
# get_tool_call_output_schema
# ---------------------------------------------------------------------------


def test_tool_output_schema():
    tool = EchoTool()
    schema = tool.get_tool_call_output_schema()
    assert "properties" in schema
    assert "output" in schema["properties"]


# ---------------------------------------------------------------------------
# Examples validation
# ---------------------------------------------------------------------------


def test_valid_examples_pass():
    # EchoTool and PersonTool should construct without error
    EchoTool()
    PersonTool()


def test_mismatched_example_type_raises():
    with pytest.raises(ValueError, match="Example"):

        class BadExampleTool(Tool[str]):
            name: ClassVar[str] = "bad_examples"
            description: ClassVar[str] = "bad"

            def _run(self, x: str) -> str:
                return x

            def get_examples(self) -> list[tuple[str, str]]:
                return [(123, "oops")]  # type: ignore[list-item]

        BadExampleTool()


# ---------------------------------------------------------------------------
# Logging context injection
# ---------------------------------------------------------------------------


def test_tool_injects_log_context():
    """Tool.__call__ should inject tool name into log context."""
    captured_context = {}

    class ContextCaptureTool(Tool[str]):
        name: ClassVar[str] = "capture"
        description: ClassVar[str] = "Captures log context."

        def _run(self, x: str) -> str:
            captured_context.update(get_log_context())
            return x

    tool = ContextCaptureTool()
    tool("test")

    assert captured_context.get("TOOL") == "capture"
    assert "TOOLCHAIN" in captured_context


def test_tool_nests_toolchain():
    """Nested tool calls should build a toolchain path."""
    inner_context = {}

    class InnerTool(Tool[str]):
        name: ClassVar[str] = "inner_tool"
        description: ClassVar[str] = "inner"

        def _run(self, x: str) -> str:
            inner_context.update(get_log_context())
            return x

    class OuterTool(Tool[str]):
        name: ClassVar[str] = "outer_tool"
        description: ClassVar[str] = "outer"
        inner: InnerTool

        def _run(self, x: str) -> str:
            return self.inner(x)

    inner = InnerTool()
    outer = OuterTool(inner=inner)
    outer("test")

    # Inner tool should have a nested toolchain (contains /)
    toolchain = inner_context.get("TOOLCHAIN", "")
    assert "/" in str(toolchain)


# ---------------------------------------------------------------------------
# _validate_run edge cases
# ---------------------------------------------------------------------------


def test_validate_run_rejects_missing_annotation():
    with pytest.raises(ValueError, match="no type annotation"):

        class NoAnnotation(Tool[str]):
            name: ClassVar[str] = "no_anno"
            description: ClassVar[str] = "bad"

            def _run(self, x) -> str:  # missing annotation on x
                return str(x)

        NoAnnotation()


def test_validate_run_rejects_missing_return():
    with pytest.raises(ValueError, match="Return type"):

        class NoReturn(Tool):
            name: ClassVar[str] = "no_return"
            description: ClassVar[str] = "bad"

            def _run(self, x: str):  # no return annotation
                return x

        NoReturn()
