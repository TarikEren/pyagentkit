# src/pyagentkit/definitions.py

from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Union

from pydantic import BaseModel, Field


class ToolReturnValue(Enum):
    """
    The return value for a tool result
    """

    success = "success"
    error = "error"
    fatal = "fatal"


class FinalResponse(BaseModel):
    type: Literal["final"]


class tool_params(BaseModel):
    """
    Tool parameters
    """

    name: str
    value: Any


class tool_call_schema(BaseModel):
    """
    Tool call object
    """

    name: str
    params: list[tool_params]


class ToolCallResponse(BaseModel):
    type: Literal["tool_call"]
    tool_call: tool_call_schema


class AgentResponse(BaseModel):
    response: Union[FinalResponse, ToolCallResponse] = Field(discriminator="type")
    message: str


# Tool types
type TypeTool = Callable[..., ToolResult]
type TypeAsyncTool = Callable[..., Awaitable[ToolResult]]

# Synchronous hooks
type TypeHookOnToolCall = Callable[[str, dict], None]
type TypeHookOnToolRetry = Callable[[str, dict, str], None]
type TypeHookOnToolSuccess = Callable[[str, dict], None]
type TypeHookOnResponse = Callable[[AgentResponse], None]
type TypeHookOnResponseRetry = Callable[[int, str, str], None]

# Asynchronous hooks
type TypeAsyncHookOnToolCall = Callable[[str, dict], Awaitable[None]]
type TypeAsyncHookOnToolSuccess = Callable[[str, dict], Awaitable[None]]
type TypeAsyncHookOnToolRetry = Callable[[str, dict, str], Awaitable[None]]
type TypeAsyncHookOnResponse = Callable[[AgentResponse], Awaitable[None]]
type TypeAsyncHookOnResponseRetry = Callable[[int, str, str], Awaitable[None]]


class AgentDependencies(BaseModel):
    """
    Basic dependencies class which other dependency classes will
    inherit from.
    """


class ToolResult(BaseModel):
    """
    Return value of a tool function
    """

    return_value: ToolReturnValue
    content: str


class BaseTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    signature: str
    desc: str
    need_deps: bool = False
    deps_param: str | None = None
    requires_approval: bool = True


class RegisteredSyncTool(BaseTool):
    function: TypeTool


class RegisteredAsyncTool(BaseTool):
    function: TypeAsyncTool | TypeTool


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            response_tokens=self.response_tokens + other.response_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
