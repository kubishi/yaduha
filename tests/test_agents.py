"""Tests for yaduha.agent implementations: OpenAIAgent, AnthropicAgent."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from yaduha.agent import AgentResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleResponse(BaseModel):
    answer: str
    confidence: float


def _openai_text_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a mock OpenAI chat completion response (text mode)."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].message.tool_calls = None
    mock.choices[0].message.model_dump_json.return_value = json.dumps(
        {"role": "assistant", "content": content}
    )
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


def _openai_parsed_response(parsed: BaseModel, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a mock OpenAI chat completion response (structured output mode)."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.parsed = parsed
    mock.choices[0].message.content = parsed.model_dump_json()
    mock.choices[0].message.tool_calls = None
    mock.choices[0].message.model_dump_json.return_value = json.dumps(
        {"role": "assistant", "content": parsed.model_dump_json()}
    )
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


def _openai_tool_call_response(tool_name: str, tool_args: dict, tool_call_id: str = "call_abc123"):
    """Build a mock OpenAI response that triggers a tool call."""
    tool_call = MagicMock()
    tool_call.type = "function"
    tool_call.id = tool_call_id
    tool_call.function.name = tool_name
    tool_call.function.arguments = json.dumps(tool_args)

    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = None
    mock.choices[0].message.tool_calls = [tool_call]
    mock.choices[0].message.model_dump_json.return_value = json.dumps(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                }
            ],
        }
    )
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    return mock


# ===========================================================================
# OpenAIAgent
# ===========================================================================


class TestOpenAIAgent:
    def test_text_response(self):
        with patch("yaduha.agent.openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _openai_text_response("hello world")

            from yaduha.agent.openai import OpenAIAgent

            agent = OpenAIAgent(model="gpt-4o", api_key="test-key")
            result = agent.get_response([{"role": "user", "content": "say hello"}])

            assert isinstance(result, AgentResponse)
            assert result.content == "hello world"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 5
            assert result.response_time > 0

    def test_structured_output(self):
        with patch("yaduha.agent.openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            expected = SimpleResponse(answer="42", confidence=0.95)
            mock_client.beta.chat.completions.parse.return_value = _openai_parsed_response(expected)

            from yaduha.agent.openai import OpenAIAgent

            agent = OpenAIAgent(model="gpt-4o", api_key="test-key")
            result = agent.get_response(
                [{"role": "user", "content": "what is the answer?"}],
                response_format=SimpleResponse,
            )

            assert isinstance(result.content, SimpleResponse)
            assert result.content.answer == "42"
            assert result.content.confidence == 0.95

    def test_tool_calling_loop(self):
        with patch("yaduha.agent.openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            # First call returns a tool call, second call returns text
            mock_client.chat.completions.create.side_effect = [
                _openai_tool_call_response("echo", {"text": "hello"}),
                _openai_text_response("done"),
            ]

            from typing import ClassVar

            from yaduha.agent.openai import OpenAIAgent
            from yaduha.tool import Tool

            class EchoTool(Tool[str]):
                name: ClassVar[str] = "echo"
                description: ClassVar[str] = "Echo text."

                def _run(self, text: str) -> str:
                    return f"echo: {text}"

            agent = OpenAIAgent(model="gpt-4o", api_key="test-key")
            result = agent.get_response(
                [{"role": "user", "content": "test"}],
                tools=[EchoTool()],
            )

            assert result.content == "done"
            assert mock_client.chat.completions.create.call_count == 2

    def test_no_content_raises(self):
        with patch("yaduha.agent.openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = _openai_text_response("")
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = None
            mock_client.chat.completions.create.return_value = mock_response

            from yaduha.agent.openai import OpenAIAgent

            agent = OpenAIAgent(model="gpt-4o", api_key="test-key")
            with pytest.raises(ValueError, match="No content"):
                agent.get_response([{"role": "user", "content": "test"}])


# ===========================================================================
# AnthropicAgent
# ===========================================================================


class TestAnthropicAgent:
    def test_convert_messages_extracts_system(self):
        from yaduha.agent.anthropic import AnthropicAgent

        agent = AnthropicAgent(model="claude-sonnet-4-5", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
        ]
        system, anthropic_msgs = agent._convert_messages(messages)

        assert system == "You are helpful."
        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"

    def test_convert_messages_handles_tool_results(self):
        from yaduha.agent.anthropic import AnthropicAgent

        agent = AnthropicAgent(model="claude-sonnet-4-5", api_key="test-key")
        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "echo: hi"},
        ]
        system, anthropic_msgs = agent._convert_messages(messages)

        assert system is None
        assert len(anthropic_msgs) == 3
        # Tool result should be wrapped in user message
        assert anthropic_msgs[2]["role"] == "user"

    def test_convert_tools(self):
        from typing import ClassVar

        from yaduha.agent.anthropic import AnthropicAgent
        from yaduha.tool import Tool

        class EchoTool(Tool[str]):
            name: ClassVar[str] = "echo"
            description: ClassVar[str] = "Echo text."

            def _run(self, text: str) -> str:
                return f"echo: {text}"

        agent = AnthropicAgent(model="claude-sonnet-4-5", api_key="test-key")
        anthropic_tools = agent._convert_tools([EchoTool()])

        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "echo"
        assert "input_schema" in anthropic_tools[0]

    def test_text_response(self):
        with patch("yaduha.agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            from anthropic.types import TextBlock

            mock_response = MagicMock()
            mock_response.content = [TextBlock(type="text", text="bonjour")]
            mock_response.usage.input_tokens = 15
            mock_response.usage.output_tokens = 3
            mock_response.model_dump.return_value = {
                "content": [{"type": "text", "text": "bonjour"}]
            }
            mock_client.messages.create.return_value = mock_response

            from yaduha.agent.anthropic import AnthropicAgent

            agent = AnthropicAgent(model="claude-sonnet-4-5", api_key="test-key")
            result = agent.get_response([{"role": "user", "content": "say hello in French"}])

            assert result.content == "bonjour"
            assert result.prompt_tokens == 15
            assert result.completion_tokens == 3

    def test_structured_output_json(self):
        with patch("yaduha.agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            from anthropic.types import TextBlock

            json_text = '{"answer": "42", "confidence": 0.95}'
            mock_response = MagicMock()
            mock_response.content = [TextBlock(type="text", text=json_text)]
            mock_response.usage.input_tokens = 20
            mock_response.usage.output_tokens = 10
            mock_response.model_dump.return_value = {
                "content": [{"type": "text", "text": json_text}]
            }
            mock_client.messages.create.return_value = mock_response

            from yaduha.agent.anthropic import AnthropicAgent

            agent = AnthropicAgent(model="claude-sonnet-4-5", api_key="test-key")
            result = agent.get_response(
                [{"role": "user", "content": "answer?"}],
                response_format=SimpleResponse,
            )

            assert isinstance(result.content, SimpleResponse)
            assert result.content.answer == "42"

    def test_structured_output_strips_code_fences(self):
        with patch("yaduha.agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            from anthropic.types import TextBlock

            fenced = '```json\n{"answer": "yes", "confidence": 0.8}\n```'
            mock_response = MagicMock()
            mock_response.content = [TextBlock(type="text", text=fenced)]
            mock_response.usage.input_tokens = 20
            mock_response.usage.output_tokens = 10
            mock_response.model_dump.return_value = {"content": [{"type": "text", "text": fenced}]}
            mock_client.messages.create.return_value = mock_response

            from yaduha.agent.anthropic import AnthropicAgent

            agent = AnthropicAgent(model="claude-sonnet-4-5", api_key="test-key")
            result = agent.get_response(
                [{"role": "user", "content": "test"}],
                response_format=SimpleResponse,
            )

            assert isinstance(result.content, SimpleResponse)
            assert result.content.answer == "yes"

    def test_no_content_raises(self):
        with patch("yaduha.agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = []  # No text blocks, no tool blocks
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 0
            mock_response.model_dump.return_value = {"content": []}
            mock_client.messages.create.return_value = mock_response

            from yaduha.agent.anthropic import AnthropicAgent

            agent = AnthropicAgent(model="claude-sonnet-4-5", api_key="test-key")
            with pytest.raises(ValueError, match="No content"):
                agent.get_response([{"role": "user", "content": "test"}])
