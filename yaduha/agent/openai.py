import time
import json
from typing import ClassVar, List, Literal, Type, overload, cast
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import Field, BaseModel

from yaduha.agent import Agent, AgentResponse, TAgentResponseContentType
from yaduha.tool import Tool


class OpenAIAgent(Agent):
    model: Literal["gpt-4o", "gpt-4o-mini", "gpt-5"]
    name: ClassVar[str] = "openai_agent"
    api_key: str = Field(..., description="The OpenAI API key.", exclude=True)
    temperature: float = Field(default=0.0, description="The temperature for the model's responses.")

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

        client = OpenAI(api_key=self.api_key)
        chat_tools = [tool.get_tool_call_schema() for tool in (tools or [])]
        tool_map = {tool.name: tool for tool in (tools or [])}

        self.log({"event": "get_response_start", "messages": messages, "tools": [tool.name for tool in (tools or [])]})

        while True:
            if response_format is str:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=chat_tools,
                    temperature=self.temperature
                )
                self.log({"event": "get_response_received", "response": response})
                msg = json.loads(response.choices[0].message.model_dump_json())
                messages.append(msg)

                if not response.choices[0].message.tool_calls:
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("No content in response")
                    self.log({"event": "get_response_content", "content": content})
                    return cast(
                        AgentResponse[TAgentResponseContentType],
                        AgentResponse(
                            content=content,
                            response_time=time.time() - start_time,
                            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                            completion_tokens=response.usage.completion_tokens if response.usage else 0,
                        )
                    )
            else:
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    tools=chat_tools,
                    response_format=response_format,
                    temperature=self.temperature
                )
                self.log({"event": "get_response_received", "response": response})
                msg = json.loads(response.choices[0].message.model_dump_json())
                messages.append(msg)

                if not response.choices[0].message.tool_calls:
                    parsed = response.choices[0].message.parsed
                    if not parsed:
                        raise ValueError("No content in response")
                    self.log({"event": "get_response_parsed", "parsed": parsed})
                    return cast(
                        AgentResponse[TAgentResponseContentType],
                        AgentResponse(
                            content=parsed,
                            response_time=time.time() - start_time,
                            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                            completion_tokens=response.usage.completion_tokens if response.usage else 0,
                        )
                    )

            for tool_call in response.choices[0].message.tool_calls or []:
                if tool_call.type == "function":
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    self.log({"event": "tool_call", "tool_name": name, "arguments": args})
                    result = tool_map[name](**args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
                    self.log({"event": "tool_result", "tool_name": name, "result": result})
