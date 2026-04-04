import re
import json
import logging
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

logger = logging.getLogger("pyagentkit")


class Agent(Generic[T]):
    """Allows easier agent creation"""

    base_system_prompt: str
    llm_name: str
    response_model: Type[T]
    instructions: str | None
    agent_name: str
    message_history: list[dict[str, str]]
    tool_retries: int
    response_retries: int
    class_tools: ClassVar[dict[str, RegisteredTool]] = {}
    instance_tools: dict[str, RegisteredTool]
    tool_try: int = 0
    dependencies: Type[AgentDependencies]
    num_ctx: int
    ollama_client: ollama.Client

    def _verify_ollama_environment(self) -> None:
        """
        Checks if the Ollama server is reachable and if the llm is downloaded.
        """
        try:
            # A simple request like list local models is a good way to verify connection
            model_list = self.ollama_client.list()
            models = []
            for entry in model_list.get("models"):
                models.append(entry.get("model"))
            if self.llm_name not in models:
                raise RuntimeError(
                    f"Model {self.llm_name} not found locally",
                    f"Run `ollama pull {self.llm_name}` to install `{self.llm_name}`",
                )

        except ConnectionError as e:
            raise RuntimeError(
                f"Failed connecting to Ollama: {e}.",
                "Please ensure Ollama is running and the host address is correct",
            )

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
        all_tools = {**self.instance_tools, **self.class_tools}
        for _, value in all_tools.items():
            sig = inspect.signature(value.function)
            params = {k: v for k, v in sig.parameters.items() if k != value.deps_param}
            clean_sig = f"({', '.join(str(p) for p in params.values())})"
            result += f"\n- {value.name} {clean_sig} | Description: {value.desc}"
        return result

    @staticmethod
    def _parse_tool(tool: TypeTool) -> RegisteredTool:
        """Helper for parsing tools"""
        name = tool.__name__
        doc = tool.__doc__
        if doc is None:
            raise RuntimeError(f"Tool `{name}` has no docstring")

        signature = inspect.signature(tool)
        params = list(signature.parameters.values())
        need_deps = (
            len(params) > 0
            and params[0].annotation is not inspect.Parameter.empty
            and isinstance(params[0].annotation, type)
            and issubclass(params[0].annotation, AgentDependencies)
        )

        return RegisteredTool(
            name=name,
            signature=str(signature),
            function=tool,
            desc=doc,
            need_deps=need_deps,
            deps_param=params[0].name if need_deps else None,
        )

    def _add_tool(self, tool: TypeTool) -> None:
        new_tool = self._parse_tool(tool)
        self.instance_tools[new_tool.name] = new_tool

    @classmethod
    def register_tool(cls, tool: TypeTool):
        """Decorator: Adds a class-wide tool"""
        new_tool = Agent._parse_tool(tool)
        cls.class_tools[new_tool.name] = new_tool
        return tool

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
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        ollama_url: str | None = None,
        tools: list[TypeTool] | None = None,
    ):
        self.base_system_prompt = system_prompt or ""
        self.llm_name = llm_name
        self.response_model = response_model
        self.instructions = instructions or ""
        self.agent_name = agent_name or self.llm_name
        self.tool_retries = tool_retries
        self.response_retries = response_retries
        self.message_history = []
        self.ollama_options: dict = {"num_ctx": num_ctx}
        if temperature:
            self.ollama_options["temperature"] = temperature
        if top_p:
            self.ollama_options["top_p"] = top_p
        if seed:
            self.ollama_options["seed"] = seed
        self.instance_tools = {}
        self.ollama_url = ollama_url
        self.ollama_client = (
            ollama.Client(host=self.ollama_url) if ollama_url else ollama.Client()
        )
        self._verify_ollama_environment()
        if tools:
            for tool in tools:
                self._add_tool(tool)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.class_tools = {}

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
        all_tools = {**self.class_tools, **self.instance_tools}
        while self.tool_try < self.tool_retries:
            tool_name = tool_call.name
            tool_params = tool_call.params
            # Find if there is such tool registered
            accepted_tool = all_tools.get(tool_name)
            if accepted_tool is None:
                raise ToolExceptionError(
                    f"Tool `{tool_call.name}` not found, check the tool name and respond with a valid tool name",
                )
            params = ""
            for param in tool_params:
                params += f"\n{param.name}: {param.value}"
            choice = (
                input(
                    f"[Info] Agent {self.agent_name} wants to call tool {tool_name} with params:\n{params}\nAllow tool call? (Y/n): "
                )
                .lower()
                .strip()
            )
            if choice not in ["y", ""]:
                logger.info("Cancelled tool call")
                self.message_history.append(
                    {
                        "role": "user",
                        "content": f"Tool `{tool_name}` with params `{tool_params}` has been rejected by the user. Generate final response telling what the user should do in order to finish the task",
                    }
                )
                self.tool_try += 1
                return

            # Handle tool arguments
            kwargs = {p.name: p.value for p in tool_params}
            if accepted_tool.need_deps:
                if self.current_deps is None:
                    raise ToolExceptionFatal(
                        f"Tool `{tool_name}` requires dependencies but none were provided to handle_response"
                    )
                if accepted_tool.deps_param is None:
                    raise ToolExceptionFatal(
                        f"Registration Error (Tool `{tool_name}` requires dependencies but has none)"
                    )
                kwargs[accepted_tool.deps_param] = self.current_deps

            # Validate kwargs against the actual function signature
            sig = inspect.signature(accepted_tool.function)
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
                    logger.warning("%s", unknown_message)
                    parts.append(unknown_message)
                if missing:
                    missing_message = f"Missing required parameters: {sorted(missing)}"
                    logger.warning("%s", missing_message)
                    parts.append(missing_message)
                valid_list = [
                    f"{name}: {inspect.formatannotation(p.annotation) if p.annotation is not inspect.Parameter.empty else 'any'}"
                    for name, p in sig.parameters.items()
                ]
                parts.append(f"Valid parameters for `{tool_name}`: {valid_list}")
                raise ToolExceptionError(". ".join(parts))

            # Call tool and get return value
            try:
                tool_return = accepted_tool.function(**kwargs)
            except TypeError as exc:
                logger.warning(
                    "Called tool `%s` with invalid arguments `%s`",
                    tool_name,
                    kwargs,
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

    def handle_response(self, prompt: str, deps: AgentDependencies | None = None) -> T:
        response_try = 0
        compiled_system_prompt = f"""{self.base_system_prompt}

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
        self.current_deps = deps
        if len(self.message_history) == 0:
            self.message_history.append(
                {"role": "system", "content": compiled_system_prompt}
            )
        self.message_history.append(
            {"role": "user", "content": prompt},
        )
        while response_try < self.response_retries:
            logger.debug("Response try: %s", response_try)
            try:
                response = self.ollama_client.chat(
                    model=self.llm_name,
                    messages=self.message_history,
                    options=self.ollama_options,
                )
                content = response.message.content
                if content is None:
                    raise RuntimeError(
                        f"Failed to get response from agent {self.agent_name}"
                    )
                logger.debug("Content from agent %s:\n%s", self.agent_name, content)

                # Append the assistant message because it doesn't know that it
                # actually did anything
                self.message_history.append({"role": "assistant", "content": content})

                validated = self.response_model.model_validate_json(content)

                logger.info("[%s]: %s", self.agent_name, validated.message)
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
