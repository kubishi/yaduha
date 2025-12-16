import time
import json
from typing import ClassVar, List, Literal, Type, overload, cast, Any
from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolUseBlock, TextBlock, ToolResultBlockParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import Field, BaseModel

from yaduha.agent import Agent, AgentResponse, TAgentResponseContentType
from yaduha.tool import Tool


class ClaudeAgent(Agent):
    model: Literal["claude-sonnet-4-5", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
    name: ClassVar[str] = "claude_agent"
    api_key: str = Field(..., description="The Claude API key.", exclude=True)
    temperature: float = Field(default=0.0, description="The temperature for the model's responses.")
    max_tokens: int = Field(default=4096, description="Maximum tokens in the response.")

    def _convert_messages(self, messages: List[ChatCompletionMessageParam]) -> tuple[str | None, List[MessageParam]]:
        """Convert OpenAI-style messages to Anthropic format.
        
        Returns (system_prompt, messages) tuple since Anthropic handles system separately.
        """
        system_prompt: str | None = None
        anthropic_messages: List[MessageParam] = []
        
        for msg in messages:
            # Cast to dict for easier access since ChatCompletionMessageParam is a TypedDict union
            msg_dict = cast(dict[str, Any], msg)
            role = msg_dict.get("role")
            content = msg_dict.get("content", "")
            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content else ""
            
            if role == "system":
                # Anthropic takes system as a separate parameter
                system_prompt = content
            elif role == "assistant":
                # Check if this was a tool use response
                tool_calls = msg_dict.get("tool_calls")
                if tool_calls:
                    # Convert to Anthropic tool_use blocks
                    content_blocks: List[Any] = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in tool_calls:
                        tc_dict = cast(dict[str, Any], tc)
                        if tc_dict.get("type") == "function":
                            func = tc_dict.get("function", {})
                            args = func.get("arguments", "{}")
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc_dict.get("id", ""),
                                "name": func.get("name", ""),
                                "input": json.loads(args) if isinstance(args, str) else args
                            })
                    anthropic_messages.append(cast(MessageParam, {"role": "assistant", "content": content_blocks}))
                else:
                    anthropic_messages.append(cast(MessageParam, {"role": "assistant", "content": content}))
            elif role == "tool":
                # Convert tool results to Anthropic format
                tool_result_block: ToolResultBlockParam = {
                    "type": "tool_result",
                    "tool_use_id": str(msg_dict.get("tool_call_id", "")),
                    "content": content
                }
                anthropic_messages.append(cast(MessageParam, {"role": "user", "content": [tool_result_block]}))
            elif role == "user":
                anthropic_messages.append(cast(MessageParam, {"role": "user", "content": content}))
        
        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: List["Tool"]) -> List[ToolParam]:
        """Convert tools to Anthropic format."""
        anthropic_tools: List[ToolParam] = []
        for tool in tools:
            schema = tool.get_tool_call_schema()
            func = schema["function"]
            anthropic_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
        return anthropic_tools

    # overload: text
    @overload
    def get_response(
        self,
        messages: List[ChatCompletionMessageParam],
        response_format: Type[str] = str,
        tools: List["Tool"] | None = None,
    ) -> AgentResponse[str]: ...
    # overload: model
    @overload
    def get_response(
        self,
        messages: List[ChatCompletionMessageParam],
        response_format: Type[TAgentResponseContentType],
        tools: List["Tool"] | None = None,
    ) -> AgentResponse[TAgentResponseContentType]: ...

    def get_response(
        self,
        messages: List[ChatCompletionMessageParam],
        response_format: Type[TAgentResponseContentType] = str,
        tools: List["Tool"] | None = None,
    ) -> AgentResponse[TAgentResponseContentType]:
        start_time = time.time()

        client = Anthropic(api_key=self.api_key)
        anthropic_tools = self._convert_tools(tools) if tools else []
        tool_map = {tool.name: tool for tool in (tools or [])}

        self.log({"event": "get_response_start", "messages": messages, "tools": [tool.name for tool in (tools or [])]})

        # Track total tokens across the conversation
        total_prompt_tokens = 0
        total_completion_tokens = 0

        while True:
            system_prompt, anthropic_messages = self._convert_messages(messages)
            
            # For structured output without tools, inject JSON instruction
            effective_system = system_prompt or ""
            if response_format is not str and not anthropic_tools and issubclass(response_format, BaseModel):
                schema_str = json.dumps(response_format.model_json_schema())
                json_instruction = f"\n\nRespond with valid JSON matching this schema: {schema_str}\nRespond ONLY with the JSON object, no markdown, no explanation."
                effective_system = (effective_system + json_instruction).strip()

            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": anthropic_messages,
                "temperature": self.temperature,
            }
            if effective_system:
                request_kwargs["system"] = effective_system
            if anthropic_tools:
                request_kwargs["tools"] = anthropic_tools

            response = client.messages.create(**request_kwargs)
            self.log({"event": "get_response_received", "response": response.model_dump()})

            # Track token usage
            total_prompt_tokens += response.usage.input_tokens
            total_completion_tokens += response.usage.output_tokens

            # Extract blocks from response
            tool_use_blocks = [block for block in response.content if isinstance(block, ToolUseBlock)]
            text_blocks = [block for block in response.content if isinstance(block, TextBlock)]

            # Build assistant message in OpenAI-compatible format
            tool_calls_for_history = []
            for block in tool_use_blocks:
                tool_calls_for_history.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
            text_content = ""
            for block in text_blocks:
                text_content += block.text
            
            # Append assistant message to history immediately (like OpenAI)
            message: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": text_content,
            }
            if tool_calls_for_history:
                message["tool_calls"] = tool_calls_for_history
            messages.append(message)

            # Check for tool use - early return if none (like OpenAI)
            if not tool_use_blocks:
                if not text_content:
                    raise ValueError("No content in response")
                
                self.log({"event": "get_response_content", "content": text_content})

                # Handle text response
                if response_format is str:
                    return cast(
                        AgentResponse[TAgentResponseContentType],
                        AgentResponse(
                            content=text_content,
                            response_time=time.time() - start_time,
                            prompt_tokens=total_prompt_tokens,
                            completion_tokens=total_completion_tokens,
                        )
                    )
                
                # Handle structured output - parse JSON
                try:
                    json_content = text_content.strip()
                    if json_content.startswith("```"):
                        lines = json_content.split("\n")
                        json_lines = []
                        in_block = False
                        for line in lines:
                            if line.startswith("```"):
                                in_block = not in_block
                                continue
                            if in_block or not line.startswith("```"):
                                json_lines.append(line)
                        json_content = "\n".join(json_lines).strip()
                    
                    parsed = response_format(**json.loads(json_content))
                except (json.JSONDecodeError, ValueError) as e:
                    self.log({"event": "parse_error", "content": text_content, "error": str(e)})
                    raise ValueError(f"Failed to parse response as {response_format.__name__}: {e}\nContent: {text_content}")

                self.log({"event": "get_response_parsed", "parsed": parsed})
                return cast(
                    AgentResponse[TAgentResponseContentType],
                    AgentResponse(
                        content=parsed,
                        response_time=time.time() - start_time,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                    )
                )

            # Handle tool calls (only reached if tool_use_blocks exist)
            for block in tool_use_blocks:
                name = block.name
                args = block.input if isinstance(block.input, dict) else {}
                self.log({"event": "tool_call", "tool_name": name, "arguments": args})
                result = tool_map[name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.id,
                    "content": str(result),
                })
                self.log({"event": "tool_result", "tool_name": name, "result": result})
