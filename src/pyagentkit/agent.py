import re
import json
import inspect
from typing import (
    ClassVar,
    Generic,
    Literal,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import ollama
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails
from .definitions import (
    AgentDependencies,
    AgentResponse,
    RegisteredTool,
    TypeTool,
    tool_call_schema,
    ToolReturnValue,
)
from .exceptions import (
    AgentExceptionError,
    AgentExceptionFatal,
    ToolExceptionError,
    ToolExceptionFatal,
)

T = TypeVar("T", bound=AgentResponse)


class Agent(Generic[T]):
    """Allows easier agent creation"""

    system_prompt: str
    llm_name: str
    response_model: Type[T]
    instructions: str | None
    agent_name: str
    message_history: list[dict[str, str]]
    tool_retries: int
    response_retries: int
    tool_registry: ClassVar[dict[str, RegisteredTool]] = {}
    tool_try: int = 0
    dependencies: Type[AgentDependencies]
    num_ctx: int

    def _build_schema_prompt(self) -> str:
        """
        Generates human-readable JSON examples from the response model.
        Handles the discriminated union (final vs tool_call) and appends
        any extra fields defined on subclasses.
        """
        # Discover extra fields added by the subclass (e.g. test: int, sub: SubResponse)
        base_fields = set(AgentResponse.model_fields.keys())
        extra_fields = {
            k: v
            for k, v in self.response_model.model_fields.items()
            if k not in base_fields
        }

        def placeholder(annotation) -> object:
            """Produce a sensible placeholder for a given type annotation."""
            origin = get_origin(annotation)
            if origin is Union:
                # Pick the first non-None arg
                args = [a for a in get_args(annotation) if a is not type(None)]
                return placeholder(args[0]) if args else None
            if origin is Literal:
                return get_args(annotation)[0]
            if annotation is str:
                return "string"
            if annotation is int:
                return "int"
            if annotation is float:
                return "float"
            if annotation is bool:
                return "boolean"
            if annotation is list or origin is list:
                inner = get_args(annotation)
                return [placeholder(inner[0])] if inner else []
            if annotation is dict or origin is dict:
                return {}
            # Nested Pydantic model - recurse
            try:
                if issubclass(annotation, BaseModel):
                    return _pydantic_example(annotation)
            except TypeError:
                pass
            return None

        def _pydantic_example(model: type[BaseModel]) -> dict:
            result = {}
            for name, field in model.model_fields.items():
                result[name] = placeholder(field.annotation)
            return result

        # Build the two canonical examples
        tool_example = {
            "response": {
                "type": "tool_call",
                "tool_call": {
                    "name": "<tool_name>",
                    "params": [{"name": "<param_name>", "value": "<param_value>"}],
                },
            },
            "message": "<why you're calling the tool>",
        }
        final_example = {
            "response": {"type": "final"},
            "message": "<your answer or result of your operation(s)>",
        }

        # Append any subclass-defined extra fields to both examples
        for key, field in extra_fields.items():
            tool_example[key] = placeholder(field.annotation)
            final_example[key] = placeholder(field.annotation)

        lines = [
            "## Response format",
            "You MUST respond with one of these two JSON shapes and nothing else.",
            "",
            "When calling a tool:",
            "```json",
            json.dumps(tool_example, indent=2),
            "```",
            "",
            "When giving a final answer:",
            "```json",
            json.dumps(final_example, indent=2),
            "```",
        ]

        return "\n".join(lines)

    def _get_tools(self):
        """Gets the tooling data for the system prompt"""
        result = "## Tools At Your Disposal"
        for _, value in self.tool_registry.items():
            result += f"\n- {value.name} {value.signature} | Description: {value.desc}"
        return result

    def __init__(
        self,
        llm_name: str,
        tool_retries: int = 3,
        response_retries: int = 3,
        system_prompt: str | None = "",
        instructions: str | None = "",
        agent_name: str | None = None,
        response_model: Type[T] = AgentResponse,
        num_ctx: int = 8192,
    ):
        self.system_prompt = system_prompt or ""
        self.llm_name = llm_name
        self.response_model = response_model
        self.instructions = instructions or ""
        self.agent_name = agent_name or self.llm_name
        self.tool_retries = tool_retries
        self.response_retries = response_retries
        self.message_history = []
        self.num_ctx = num_ctx

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.tool_registry = {}

    @classmethod
    def register_tool(cls, tool: TypeTool):
        """Decorator: Adds a tool to the tool_registry list"""
        name = tool.__name__

        if tool.__doc__ is None:
            raise RuntimeError(f"Tool `{name}` has no docstring")

        signature = inspect.signature(tool)

        cls.tool_registry[name] = RegisteredTool(
            name=name, signature=str(signature), function=tool, desc=tool.__doc__
        )
        return tool

    def _print_validation_errors(self, errors: list[ErrorDetails]) -> str:
        """
        Returns the ValidationError errors in a readable message format.
        Used to create readable messages for the agent

        Args:
            errors (list[ErrorDetails]): Validation error details
        Returns:
            (str): Printable message
        """
        result = "Your response failed validation, handle the following issues and generate your answer with the same `type`\nEncountered errors:"
        for error in errors:
            result += f"\nType: {error['type']}"
            result += f"\nField: {'.'.join(map(str, error['loc']))}"
            result += f"\nError message: {error['msg']}"
        result += "\nCRITICAL: Fix the errors and respond ONLY with valid JSON. Make sure you close out your curly braces ('{'). Do NOT include any markdown formatting."
        return result

    def _get_tool_from_registry(self, func_name: str) -> TypeTool | None:
        """
        Checks the tool registry list and finds if there's a matching tool to the given
        tool name
        Args:
            func_name (str): Name of the tool to search for
        Returns:
            (TypeTool | None): The tool itself or None
        """
        # Check if the tool is in the registry
        entry = self.tool_registry.get(func_name)
        if entry is not None:
            return entry.function

    def _strip_markdown_formatting(self, message: str) -> str:
        """
        Strips the (```) from the provided message
        Args:
            message (str): Message to trim
        Returns:
            (str): Stripped message
        """
        raw = message.strip()
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            raw = match.group(1).strip() if match else raw
        return raw

    def _validate_agent_logic(
        self,
        response: AgentResponse,
    ) -> None:
        """
        Hook for validating agent-specific logic
        Raise AgentExceptionError on fail
        """
        pass

    def _handle_tool_call(self, tool_call: tool_call_schema) -> None:
        while self.tool_try < self.tool_retries:
            tool_name = tool_call.name
            tool_params = tool_call.params
            # Find if there is such tool registered
            accepted_tool = self._get_tool_from_registry(tool_name)
            if accepted_tool is None:
                raise ToolExceptionError(
                    f"Tool `{tool_call.name}` not found, check the tool name and respond with a valid tool name",
                )
            print(
                f"\n[Info]: Agent {self.agent_name} wants to call tool {tool_name} with params:"
            )
            for param in tool_params:
                print("-", param)
            choice = input("Allow tool call? (Y/n): ").lower().strip()
            if choice not in ["y", ""]:
                print("[Info]: Cancelled tool call")
                self.message_history.append(
                    {
                        "role": "user",
                        "content": f"Tool `{tool_name}` with params `{tool_params}` has been rejected by the user. Generate final response saying that the user has rejected the tool call",
                    }
                )
                self.tool_try += 1
                return

            # Handle tool arguments
            kwargs = {p.name: p.value for p in tool_params}

            # Validate kwargs against the actual function signature
            sig = inspect.signature(accepted_tool)
            valid_params = set(sig.parameters.keys())
            provided_params = set(kwargs.keys())

            unknown = provided_params - valid_params
            missing = {
                name
                for name, param in sig.parameters.items()
                if param.default is inspect.Parameter.empty
                and name not in provided_params
            }

            if unknown or missing:
                parts = []
                if unknown:
                    unknown_message = f"Unknown parameters {sorted(unknown)}"
                    print(f"[ERROR] {unknown_message}")
                    parts.append(unknown_message)
                if missing:
                    missing_message = f"Missing required parameters: {sorted(missing)}"
                    print(f"[ERROR] {missing_message}")
                    parts.append(missing_message)
                valid_list = [
                    f"{name}: {inspect.formatannotation(p.annotation) if p.annotation is not inspect.Parameter.empty else 'any'}"
                    for name, p in sig.parameters.items()
                ]
                parts.append(f"Valid parameters for `{tool_name}`: {valid_list}")
                raise ToolExceptionError(". ".join(parts))

            # Call tool and get return value
            try:
                tool_return = accepted_tool(**kwargs)
            except TypeError as exc:
                print(
                    f"[ERROR]: Called tool `{tool_name}` with invalid arguments `{kwargs}`"
                )
                raise ToolExceptionError(
                    f"Tool `{tool_name}` call failed with invalid arguments: {exc}. "
                    f"Check parameter names and types against the tool signature."
                )
            tool_return_val = tool_return.return_value
            tool_content = tool_return.content
            if tool_return_val == ToolReturnValue.fatal:
                raise ToolExceptionFatal(tool_content)
            elif tool_return_val == ToolReturnValue.error:
                self.message_history.append(
                    {
                        "role": "user",
                        "content": f"Tool Result: ERROR\nTool Message: {tool_content}\nRead the tool message and act accordingly",
                    }
                )
                raise ToolExceptionError(tool_content)
            elif tool_return_val == ToolReturnValue.success:
                self.message_history.append(
                    {
                        "role": "user",
                        "content": f"""Tool Result: SUCCESS
Tool Response: {tool_content}

If task is done, generate `final` response and stop.""",
                    }
                )
                self.tool_try = 0
                return
        raise ToolExceptionFatal(
            f"[FATAL]: Agent {self.agent_name} has failed to generate successful tool call in {self.tool_retries} tries"
        )

    def handle_response(self, prompt: str) -> T:
        response_try = 0
        # Send prompt to agent
        # print(f"[DEBUG]: System prompt: {self.message_history[0]}")
        self.system_prompt = f"""{self.system_prompt}

{self._build_schema_prompt()}

{self._get_tools()}

## Crucial Rules - Rules To Abide By
- DON'T include any explanations, introductions or apologies.
- DON'T explain the process, execute it.
- Make sure your responses are perfect JSON objects (No missing braces, commas or quotes)
- Make sure your responses are matching with the schemas you've been given (No missing or invalid fields)
- DO NOT USE ``` CODE BLOCKS OR \"\"\" MULTI-LINE STRINGS, ONLY SINGLE LINE STRINGS ARE ALLOWED.
- Do NOT use placeholder values for any function parameter or file content.
- ONLY use paths that are in the format of `./target/path` or `target/path`
- DO NOT USE ANY NON-EXISTING TOOLS. DON'T MAKE UP TOOL NAMES.
"""
        self.message_history.append({"role": "system", "content": self.system_prompt})
        self.message_history.append(
            {"role": "user", "content": prompt},
        )
        while response_try < self.response_retries:
            # print(f"[DEBUG]: Last Message: {self.message_history[-1]}")
            # print(f"[DEBUG]: Response Try: {response_try}, Tool Try: {self.tool_try}")
            try:
                response = ollama.chat(
                    model=self.llm_name,
                    messages=self.message_history,
                    options={"num_ctx": self.num_ctx},
                )
                content = response.message.content
                print(f"[DEBUG] Content:\n{content}")
                if content is None:
                    raise RuntimeError(
                        f"Failed to get response from agent {self.agent_name}"
                    )

                # Append the assistant message because it doesn't know that it
                # actually did anything
                self.message_history.append({"role": "assistant", "content": content})

                # Get rid of markdown fences because some models are stupid
                # stripped = self._strip_markdown_formatting(message=content)
                # print(f"[DEBUG] Stripped: {stripped}")
                # validated = self.response_model.model_validate_json(stripped)

                validated = self.response_model.model_validate_json(content)
                # print(f"[DEBUG] Validated Response:\n{validated}\n")
                # If the response is "done", return
                print(f"[{self.agent_name}]: {validated.message}")
                self._validate_agent_logic(response=validated)
                if validated.response.type == "final":
                    return validated
                if validated.response.type == "tool_call":
                    response_try = 0
                    tool_call = validated.response.tool_call
                    # Check if a tool call is present in a `tool_call` typed response
                    if tool_call:
                        self._handle_tool_call(tool_call=tool_call)

            except ValidationError as exc:
                print(
                    f"[DEBUG] Validation error string: {self._print_validation_errors(exc.errors())}"
                )
                self.message_history.append(
                    {
                        "role": "user",
                        "content": self._print_validation_errors(exc.errors()),
                    }
                )
                response_try += 1
            except AgentExceptionError as exc:
                self.message_history.append({"role": "user", "content": exc.message})
                response_try += 1
            except ToolExceptionError as exc:
                self.message_history.append({"role": "user", "content": exc.message})
                self.tool_try += 1
            except (AgentExceptionFatal, ToolExceptionFatal) as exc:
                self.tool_try = 0
                raise RuntimeError(exc.message)
            except Exception as exc:
                self.tool_try = 0
                raise RuntimeError(f"Unhandled exception: {str(exc)}")

        raise RuntimeError(
            f"[ERROR] Agent {self.agent_name} failed to provide a valid response after {self.response_retries} attempts."
        )
