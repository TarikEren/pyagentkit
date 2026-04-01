"""Contains enums, schemas, types and global values"""

from enum import Enum
from typing import Any, Callable, Literal, Union

from pydantic import BaseModel, Field


class ToolReturnValue(Enum):
    """
    The return value for a tool result
    """

    success = "success"
    error = "error"
    fatal = "fatal"


type TypeTool = Callable[..., ToolResult]


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


class FinalResponse(BaseModel):
    type: Literal["final"]


class ToolCallResponse(BaseModel):
    type: Literal["tool_call"]
    tool_call: tool_call_schema


class AgentResponse(BaseModel):
    response: Union[FinalResponse, ToolCallResponse] = Field(discriminator="type")
    message: str


class AgentDependencies(BaseModel):
    """
    Basic dependencies class which other dependency classes will
    inherit from.
    """

    prompt: str = Field(description="The prompt to be used")


class ToolResult(BaseModel):
    """
    Return value of a tool function
    """

    return_value: ToolReturnValue
    content: str


class RegisteredTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    signature: str
    desc: str
    function: TypeTool
    need_deps: bool = False
    deps_param: str | None = None


class RegisteredCommand(BaseModel):
    """
    Defines a command which can be called by 'execute_command'
    """

    name: str = Field(description="Name of the registered command")
    accepted_args: list[str] = Field(description="Accepted arguments or flags")
    # True if non-flag args are supposed to be treated as paths
    accepts_file_path: bool = False
