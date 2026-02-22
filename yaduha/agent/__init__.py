from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, overload

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from yaduha.logger import Logger, get_global_logger

if TYPE_CHECKING:
    from yaduha.tool import Tool

TStringType = TypeVar("TStringType", bound=str)
TAgentResponseContentType = TypeVar("TAgentResponseContentType", bound=(BaseModel | str))


class AgentResponse(BaseModel, Generic[TAgentResponseContentType]):
    content: TAgentResponseContentType = Field(
        ..., description="The content of the agent's response."
    )
    response_time: float = Field(..., description="The time taken to generate the response.")
    prompt_tokens: int = Field(0, description="The number of prompt tokens used in the response.")
    completion_tokens: int = Field(
        0, description="The number of completion tokens used in the response."
    )


class Agent(BaseModel, Generic[TStringType]):
    model: TStringType
    name: ClassVar[str] = Field(..., description="The name of the agent.")
    logger: Logger = Field(
        default_factory=get_global_logger, description="The logger to use for logging."
    )

    def log(self, data: dict[str, Any]) -> None:
        self.logger.log(data={**data, "agent_model": self.model, "agent_name": self.name})

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

    @abstractmethod
    def get_response(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[TAgentResponseContentType] = str,
        tools: list["Tool"] | None = None,
    ) -> AgentResponse[TAgentResponseContentType]:
        raise NotImplementedError
