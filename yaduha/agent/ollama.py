import json
import time
from typing import Any, ClassVar, cast, overload

import requests
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from yaduha.agent import Agent, AgentResponse, TAgentResponseContentType
from yaduha.tool import Tool


class OllamaAgent(Agent):
    """Agent that uses Ollama for local LLM inference.

    Supports any model available in Ollama (e.g., llama3.2:3b, qwen2.5:7b, mistral:7b).

    For structured outputs, uses Ollama's native grammar-constrained generation via
    the 'format' parameter, which guarantees valid JSON matching the schema.

    For tool calling, requires a model that supports function calling (e.g., llama3.1, qwen2.5).
    Tool calling uses the OpenAI-compatible API endpoint.
    """

    model: str
    name: ClassVar[str] = "ollama_agent"
    base_url: str = Field(
        default="http://localhost:11434",
        description="The base URL for the Ollama server.",
    )
    temperature: float = Field(
        default=0.0,
        description="The temperature for the model's responses. Lower is more deterministic.",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in the response.",
    )

    def _get_client(self) -> OpenAI:
        """Create an OpenAI client configured for Ollama's OpenAI-compatible API."""
        return OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",  # Required by OpenAI client but ignored by Ollama
        )

    def _call_native_api(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[BaseModel],
    ) -> tuple[str, int, int]:
        """Call Ollama's native API with grammar-constrained structured output.

        Returns (content, prompt_tokens, completion_tokens).
        """
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                ollama_messages.append({"role": "system", "content": str(content)})
            elif role == "user":
                ollama_messages.append({"role": "user", "content": str(content)})
            elif role == "assistant":
                ollama_messages.append({"role": "assistant", "content": str(content)})
            # Skip tool messages for native API

        # Get the JSON schema for grammar-constrained generation
        schema = response_format.model_json_schema()

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "format": schema,  # Ollama uses this for grammar-constrained generation
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")
        # Ollama native API returns different token fields
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return content, prompt_tokens, completion_tokens

    def _inject_json_schema(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[BaseModel],
        has_tools: bool = False,
    ) -> list[ChatCompletionMessageParam]:
        """Inject JSON schema instruction into the system prompt."""
        schema = response_format.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        if has_tools:
            # When tools are present, we need to be very clear about the distinction
            json_instruction = (
                f"\n\n## IMPORTANT: TOOLS vs FINAL RESPONSE\n"
                f"You have access to tools. Use them by making tool calls with the correct arguments.\n"
                f"When you are DONE using tools and ready to give your FINAL answer, "
                f"respond with a JSON object (NOT a tool call) matching this schema:\n\n"
                f"```json\n{schema_str}\n```\n\n"
                f"CRITICAL RULES:\n"
                f"1. Tool calls require the tool's specific arguments (e.g., 'text' for translator tools)\n"
                f"2. Your FINAL response must be a JSON object matching the schema above\n"
                f"3. Do NOT put your final response inside a tool call\n"
                f"4. Do NOT include schema definitions ($defs, $ref) in your response"
            )
        else:
            json_instruction = (
                f"\n\n## RESPONSE FORMAT INSTRUCTIONS\n"
                f"You must respond with a valid JSON object. Do NOT include the schema definition in your response.\n"
                f"Generate actual data values that conform to this schema:\n\n```json\n{schema_str}\n```\n\n"
                f"IMPORTANT RULES:\n"
                f"1. Output ONLY the JSON data object - no schema definitions, no $defs, no $ref\n"
                f"2. Do NOT echo back the schema - generate actual values\n"
                f"3. No markdown code fences in your response\n"
                f"4. No explanatory text before or after the JSON"
            )

        messages = list(messages)  # Make a copy

        # Find and modify system message, or prepend one
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                original_content = msg.get("content", "")
                messages[i] = {
                    "role": "system",
                    "content": str(original_content) + json_instruction,
                }
                return messages

        # No system message found, prepend one
        messages.insert(0, {"role": "system", "content": json_instruction.strip()})
        return messages

    def _parse_json_response(self, content: str, response_format: type[BaseModel]) -> BaseModel:
        """Parse JSON from model response, handling markdown code blocks."""
        json_content = content.strip()

        # Handle markdown code fences
        if json_content.startswith("```"):
            lines = json_content.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            json_content = "\n".join(json_lines).strip()

        return response_format(**json.loads(json_content))

    @overload
    def get_response(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[str] = str,
        tools: list["Tool"] | None = None,
    ) -> AgentResponse[str]: ...

    @overload
    def get_response(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[TAgentResponseContentType],
        tools: list["Tool"] | None = None,
    ) -> AgentResponse[TAgentResponseContentType]: ...

    def get_response(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[TAgentResponseContentType] = str,
        tools: list["Tool"] | None = None,
    ) -> AgentResponse[TAgentResponseContentType]:
        start_time = time.time()

        self.log(
            {
                "event": "get_response_start",
                "messages": messages,
                "tools": [tool.name for tool in (tools or [])],
            }
        )

        # For structured output WITHOUT tools, use Ollama's native API with
        # grammar-constrained generation. This guarantees valid JSON.
        if response_format is not str and not tools and issubclass(response_format, BaseModel):
            response_format_model = cast(type[BaseModel], response_format)
            try:
                content, prompt_tokens, completion_tokens = self._call_native_api(
                    messages, response_format_model
                )
                self.log({"event": "get_response_content", "content": content})

                parsed = response_format_model(**json.loads(content))
                self.log({"event": "get_response_parsed", "parsed": parsed})

                return cast(
                    AgentResponse[TAgentResponseContentType],
                    AgentResponse(
                        content=parsed,
                        response_time=time.time() - start_time,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    ),
                )
            except (json.JSONDecodeError, ValueError, requests.RequestException) as e:
                self.log({"event": "native_api_error", "error": str(e)})
                raise ValueError(f"Failed to get structured response: {e}")

        # For text responses or when tools are involved, use OpenAI-compatible API
        client = self._get_client()
        chat_tools = [tool.get_tool_call_schema() for tool in (tools or [])]
        tool_map = {tool.name: tool for tool in (tools or [])}

        working_messages = list(messages)
        use_json_mode = False

        # When tools are present with structured output, fall back to prompt injection
        if response_format is not str and tools and issubclass(response_format, BaseModel):
            working_messages = self._inject_json_schema(
                working_messages, response_format, has_tools=True
            )
            use_json_mode = True

        # Track total tokens across the conversation
        total_prompt_tokens = 0
        total_completion_tokens = 0

        while True:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": working_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            if chat_tools:
                request_kwargs["tools"] = chat_tools

            if use_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**request_kwargs)
            self.log({"event": "get_response_received", "response": response})

            # Track token usage
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens

            # Append assistant message to history
            msg = json.loads(response.choices[0].message.model_dump_json())
            working_messages.append(msg)

            # Check for tool calls
            if not response.choices[0].message.tool_calls:
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("No content in response")

                self.log({"event": "get_response_content", "content": content})

                # Handle text response
                if response_format is str:
                    return cast(
                        AgentResponse[TAgentResponseContentType],
                        AgentResponse(
                            content=content,
                            response_time=time.time() - start_time,
                            prompt_tokens=total_prompt_tokens,
                            completion_tokens=total_completion_tokens,
                        ),
                    )

                # Handle structured output - parse JSON
                # At this point we know response_format is a BaseModel subclass (not str)
                response_format_model = cast(type[BaseModel], response_format)
                try:
                    parsed = self._parse_json_response(content, response_format_model)
                except (json.JSONDecodeError, ValueError) as e:
                    self.log({"event": "parse_error", "content": content, "error": str(e)})
                    raise ValueError(
                        f"Failed to parse response as {response_format_model.__name__}: {e}\nContent: {content}"
                    )

                self.log({"event": "get_response_parsed", "parsed": parsed})
                return cast(
                    AgentResponse[TAgentResponseContentType],
                    AgentResponse(
                        content=parsed,
                        response_time=time.time() - start_time,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                    ),
                )

            # Handle tool calls
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.type == "function":
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    self.log({"event": "tool_call", "tool_name": name, "arguments": args})
                    try:
                        result = tool_map[name](**args)
                    except TypeError as e:
                        # Model provided incorrect arguments - log and re-raise with more context
                        self.log(
                            {
                                "event": "tool_call_error",
                                "tool_name": name,
                                "arguments": args,
                                "error": str(e),
                            }
                        )
                        raise TypeError(
                            f"Tool '{name}' called with incorrect arguments. "
                            f"Received: {args}. Error: {e}"
                        ) from e
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
                    self.log({"event": "tool_result", "tool_name": name, "result": result})
