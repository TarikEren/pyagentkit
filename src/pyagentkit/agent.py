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
    ToolResult,
    TypeHookOnResponse,
    TypeHookOnResponseRetry,
    TypeHookOnToolCall,
    TypeHookOnToolRetry,
    RegisteredSyncTool,
    TokenUsage,
    TypeHookOnToolSuccess,
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
    """
    Creates base sync agent
    """

    # Base, plain system prompt
    base_system_prompt: str

    # Name of the LLM in Ollama
    llm_name: str

    # Type of response the agent will produce
    response_model: Type[T]

    # Instructions to append at the end of prompts
    instructions: str | None

    # Name for the agent (LLM name will be used if none provided)
    agent_name: str

    # Message history for the agent to view
    message_history: list[dict[str, str]]

    # Retry limit for tool calls
    tool_retries: int

    # Retry limit for responses
    response_retries: int

    # Tools registered to the entire class
    class_tools: ClassVar[dict[str, RegisteredSyncTool]] = {}

    # Tools registered to the instance only
    instance_tools: dict[str, RegisteredSyncTool]

    # Dependencies for the agent to utilize
    dependencies: Type[AgentDependencies]

    # Context limit
    num_ctx: int

    # Custom ollama client
    ollama_client: ollama.Client

    # Past message limit
    max_history: int | None = None

    # Token usage object for monitoring usage
    token_usage: TokenUsage

    # Logger object for modular logging
    logger: logging.Logger

    # Agent registry for viewing what agents are registered for the class
    _agent_registry: ClassVar[dict[str, "Agent"]] = {}

    # --- Hooks ---
    on_tool_call: TypeHookOnToolCall | None
    on_tool_retry: TypeHookOnToolRetry | None
    on_tool_success: TypeHookOnToolSuccess | None
    on_response: TypeHookOnResponse | None
    on_response_retry: TypeHookOnResponseRetry | None

    def _verify_ollama_environment(self) -> None:
        """
        Checks if the Ollama server is reachable and if the llm is downloaded.
        """
        try:
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
        # Get base fields from base response class
        base_fields = set(AgentResponse.model_fields.keys())

        # Get extra fields from custom response class
        extra_fields = {
            k: v
            for k, v in self.response_model.model_fields.items()
            if k not in base_fields
        }

        def placeholder(annotation) -> object:
            """Produce a sensible placeholder for a given type annotation."""
            origin = get_origin(annotation)
            if origin is Union:
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

        # Canonical examples
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
            "- When calling a tool:",
            "```json",
            json.dumps(tool_example, indent=2),
            "```",
            "",
            "- When giving a final answer:",
            "```json",
            json.dumps(final_example, indent=2),
            "```",
            "Do NOT use markdown formatting (```) when creating your responses AND messages",
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
    def _parse_tool(tool: TypeTool, requires_approval: bool) -> RegisteredSyncTool:
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

        return RegisteredSyncTool(
            name=name,
            signature=str(signature),
            function=tool,
            desc=doc,
            need_deps=need_deps,
            deps_param=params[0].name if need_deps else None,
            requires_approval=requires_approval,
        )

    def add_tool(self, tool: TypeTool, requires_approval: bool = True) -> None:
        new_tool = self._parse_tool(tool=tool, requires_approval=requires_approval)
        self.instance_tools[new_tool.name] = new_tool

    @classmethod
    def register_tool(
        cls, tool: TypeTool | None = None, requires_approval: bool = True
    ):
        """Decorator: Adds a class-wide tool"""

        def _decorator(t: TypeTool):
            new_tool = Agent._parse_tool(tool=t, requires_approval=requires_approval)
            cls.class_tools[new_tool.name] = new_tool
            return t

        if tool is not None:
            return _decorator(tool)

        return _decorator

    def as_tool(
        self, description: str | None, deps: AgentDependencies | None
    ) -> TypeTool:
        agent = self

        def run_agent(prompt: str) -> ToolResult:
            try:
                response = agent.handle_response(prompt=prompt, deps=deps)
                return ToolResult(
                    return_value=ToolReturnValue.success, content=response.message
                )
            except RuntimeError as err:
                return ToolResult(return_value=ToolReturnValue.error, content=str(err))

        run_agent.__name__ = agent.agent_name
        run_agent.__doc__ = (
            description or f"Run agent {agent.agent_name} with the given prompt"
        )
        return run_agent

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
        max_history: int | None = None,
        log_level: int = logging.INFO,
        on_tool_call: TypeHookOnToolCall | None = None,
        on_tool_retry: TypeHookOnToolRetry | None = None,
        on_tool_success: TypeHookOnToolSuccess | None = None,
        on_response: TypeHookOnResponse | None = None,
        on_response_retry: TypeHookOnResponseRetry | None = None,
        think: bool = False,
    ):
        self.base_system_prompt = system_prompt or ""
        self.llm_name = llm_name
        self.response_model = response_model
        self.instructions = instructions or ""
        self.agent_name = agent_name or self.llm_name
        if self.agent_name in Agent._agent_registry.keys():
            raise ValueError(
                f"An agent with name `{self.agent_name}` is already registered. Use a unique agent_name or call `.dispose()` on the existing agent first. Registered agents: {list(Agent._agent_registry.keys())}"
            )
        self.response_retries = response_retries
        self.tool_retries = tool_retries

        self.message_history = []
        self.max_history = max_history

        # Initialize the logger with the agent name
        self.logger = logging.getLogger(f"pyagentkit.{self.agent_name}")
        self.logger.setLevel(log_level)

        # Don't use root logger
        self.logger.propagate = False

        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(name)s | %(levelname)s | %(message)s")
        )
        self.logger.addHandler(handler)

        # Create Ollama options dictionary
        self.ollama_options: dict = {"num_ctx": num_ctx}
        if temperature:
            self.ollama_options["temperature"] = temperature
        if top_p:
            self.ollama_options["top_p"] = top_p
        if seed:
            self.ollama_options["seed"] = seed

        # Create custom Ollama client
        self.ollama_url = ollama_url
        self.ollama_client = (
            ollama.Client(host=self.ollama_url) if ollama_url else ollama.Client()
        )

        # Verify environment
        self._verify_ollama_environment()

        # Register instance tools
        self.instance_tools = {}
        if tools:
            for tool in tools:
                self.add_tool(tool)

        # Create TokenUsage object for token monitoring
        self.token_usage = TokenUsage()

        # Add the agent to the agent registry
        Agent._agent_registry[self.agent_name] = self

        # Initialize hooks
        self.on_tool_call = on_tool_call
        self.on_tool_retry = on_tool_retry
        self.on_tool_success = on_tool_success
        self.on_response = on_response
        self.on_response_retry = on_response_retry

        self.think = think

    def dispose(self) -> None:
        """Unregisters the agent and cleans up its logger"""
        Agent._agent_registry.pop(self.agent_name, None)
        self.logger.handlers.clear()

        # Detach logger from agent
        self.logger.propagate = False

    def __init_subclass__(cls, **kwargs) -> None:
        """Initializes subclasses"""
        # Initialize class tools
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
        Strips the (```) from the provided text
        Args:
            message (str): Message to trim
        Returns:
            (str): Stripped message
        """
        raw = message.strip()
        if "```" in raw:
            return re.sub(r"^```.*?\\n|```$", "", raw, re.DOTALL)
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

    def _on_tool_call(self, tool_name: str, params: dict) -> None:
        if self.on_tool_call:
            self.on_tool_call(tool_name, params)

    def _on_tool_retry(self, tool_name: str, params: dict, error_message: str) -> None:
        if self.on_tool_retry:
            self.on_tool_retry(tool_name, params, error_message)

    def _on_tool_success(self, tool_name: str, params: dict) -> None:
        if self.on_tool_success:
            self.on_tool_success(tool_name, params)

    def _on_response(self, response: AgentResponse) -> None:
        if self.on_response:
            self.on_response(response)

    def _on_response_retry(
        self, response_try: int, response_str: str, error_message: str
    ) -> None:
        if self.on_response_retry:
            self.on_response_retry(response_try, response_str, error_message)

    def _handle_tool_call(
        self, tool_call: tool_call_schema, deps: AgentDependencies | None
    ) -> bool:
        """
        Handles the provided tool call in self.tool_retries iterations

        Args:
            tool_call (tool_call_schema): The tool call to handle
        Returns:
            (bool): Tool result (True on success, False on reject)
        Raises:
            ToolExceptionError
            ToolExceptionFatal
        """
        all_tools = {**self.class_tools, **self.instance_tools}

        # Handle tool arguments
        tool_name = tool_call.name
        tool_params = tool_call.params
        kwargs = {p.name: p.value for p in tool_params}

        # Find if there is such tool registered
        accepted_tool = all_tools.get(tool_name)
        if accepted_tool is None:
            self._on_tool_retry(
                tool_name=tool_name,
                params=kwargs,
                error_message=f"Tool {tool_call.name} not found",
            )
            raise ToolExceptionError(
                f"Tool `{tool_call.name}` not found, check the tool name and respond with a valid tool name",
            )
        if accepted_tool.requires_approval is True:
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
                self.logger.info("Cancelled tool call")
                self.message_history.append(
                    {
                        "role": "user",
                        "content": f"Tool `{tool_name}` with params `{tool_params}` has been rejected by the user. Generate final response telling what the user should do in order to finish the task",
                    }
                )
                return False

        if accepted_tool.need_deps:
            if deps is None:
                raise ToolExceptionFatal(
                    f"Tool `{tool_name}` requires dependencies but none were provided to handle_response"
                )
            if accepted_tool.deps_param is None:
                raise ToolExceptionFatal(
                    f"Registration Error (Tool `{tool_name}` requires dependencies but has none)"
                )
            kwargs[accepted_tool.deps_param] = deps

        # Validate kwargs against the actual function signature
        sig = inspect.signature(accepted_tool.function)
        valid_params = set(sig.parameters.keys())
        provided_params = set(kwargs.keys())

        unknown = provided_params - valid_params
        missing = {
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty and name not in provided_params
        }

        if unknown or missing:
            parts = []
            if unknown:
                unknown_message = f"Unknown parameters {sorted(unknown)}"
                self.logger.warning("%s", unknown_message)
                parts.append(unknown_message)
            if missing:
                missing_message = f"Missing required parameters: {sorted(missing)}"
                self.logger.warning("%s", missing_message)
                parts.append(missing_message)
            valid_list = [
                f"{name}: {inspect.formatannotation(p.annotation) if p.annotation is not inspect.Parameter.empty else 'any'}"
                for name, p in sig.parameters.items()
            ]
            parts.append(f"Valid parameters for `{tool_name}`: {valid_list}")
            invalid_param_message = ". ".join(parts)
            self._on_tool_retry(
                tool_name=tool_name,
                params=kwargs,
                error_message=invalid_param_message,
            )
            raise ToolExceptionError(invalid_param_message)

        # Call tool and get return value
        self._on_tool_call(tool_name=tool_name, params=kwargs)
        try:
            tool_return = accepted_tool.function(**kwargs)
        except TypeError as exc:
            self.logger.warning(
                "Called tool `%s` with invalid arguments `%s`",
                tool_name,
                kwargs,
            )
            self._on_tool_retry(
                tool_name=tool_name, params=kwargs, error_message="Invalid arguments"
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
            tool_result_message = f"Tool Result: ERROR\nTool Message: {tool_content}"
            self.message_history.append(
                {
                    "role": "user",
                    "content": f"{tool_result_message}\nRead the tool message and act accordingly",
                }
            )
            self._on_tool_retry(
                tool_name=tool_name, params=kwargs, error_message=tool_result_message
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
            self._on_tool_success(tool_name=tool_name, params=kwargs)
            return True
        raise ToolExceptionFatal(
            f"[FATAL]: Agent {self.agent_name} has failed to generate successful tool call in {self.tool_retries} tries"
        )

    def clear_history(self) -> None:
        """Clear the entire message history"""
        self.message_history = []

    def save_history(self, path: str) -> None:
        """Save the message history to the given path"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.message_history, f, indent=2)
        self.logger.info("Saved message history to %s", path)

    def load_history(self, path: str) -> None:
        """Load the message history from the given JSON file"""
        acceptable_roles = ["user", "system", "assistant"]
        with open(path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            for entry in saved_data:
                if (
                    entry.get("role") not in acceptable_roles
                    or entry.get("content") is None
                ):
                    self.logger.warning(
                        "File %s contains invalid message data, failed to load messages",
                        str(path),
                    )
                    return
            self.message_history = saved_data

        self.logger.info("Loaded message history from %s", path)

    def _trim_history(self) -> None:
        """Trim the message history down if max_history param is not None"""
        if self.max_history is None:
            return
        system_messages = [m for m in self.message_history if m.get("role") == "system"]
        other_messages = [m for m in self.message_history if m.get("role") != "system"]
        if len(other_messages) > self.max_history:
            other_messages = other_messages[-self.max_history :]

            # If the user message preceeding the assistant message is removed (Assistant message is orphaned)
            # remove the assistant message
            if other_messages and other_messages[0].get("role") == "assistant":
                other_messages = other_messages[1:]

        self.message_history = system_messages + other_messages

    def handle_response(self, prompt: str, deps: AgentDependencies | None = None) -> T:
        """
        Handles response creation for agent

        Args:
            prompt (str): Prompt to work on
            deps (AgentDependencies | None): Dependencies to use (None by default)
        Returns:
            T (Generic[AgentResponse]): Response object
        Raises:
            RuntimeError
        """
        response_try = 0
        tool_try = 0
        compiled_system_prompt = f"""{self.base_system_prompt}

# Role
You are an agent that performs tool-based tasks and returns only valid JSON. Check the tools and their descriptions, call tools only if necessary.

# Goal
Complete the user task using the available tools and schemas.

# Critical Rules
- No explanations, introductions, or apologies.
- Output must be valid JSON only.
- Do not invent tools.
- Do not use placeholder values.
- Use only allowed path formats.
- Code responses must be in a single line, no markdown formatting (```) and no multiline strings (\"\"\").

# Tool Rules
- Only call tools listed here.

{self._get_tools()}

- Follow each tool schema exactly.
- If no tool applies, return a JSON error object.

# Output Contract

{self._build_schema_prompt()}

- Return exactly one JSON object.
- Match the provided schema.
- No markdown, no code fences, no extra text.
- No markdown formatting (```), no multiline strings (\"\"\") and no extra text.

# Fallback Behavior
- If required information is missing, return a structured error JSON object.
"""
        self.logger.debug(
            "All tooling for agent %s: %s\n",
            self.agent_name,
            str({**self.class_tools, **self.instance_tools}),
        )
        if len(self.message_history) == 0:
            self.message_history.append(
                {"role": "system", "content": compiled_system_prompt}
            )
        self.message_history.append(
            {
                "role": "user",
                "content": f"{prompt}\n{self.instructions}"
                if self.instructions
                else prompt,
            },
        )
        while response_try < self.response_retries and tool_try < self.tool_retries:
            self.logger.debug("Response try: %s", response_try)
            content = ""
            try:
                self._trim_history()
                response = self.ollama_client.chat(
                    model=self.llm_name,
                    messages=self.message_history,
                    options=self.ollama_options,
                    think=self.think,
                )
                _prompt_tokens = response.prompt_eval_count or 0
                _response_tokens = response.eval_count or 0
                self.token_usage += TokenUsage(
                    prompt_tokens=_prompt_tokens,
                    response_tokens=_response_tokens,
                    total_tokens=_prompt_tokens + _response_tokens,
                )
                if self.think:
                    self.logger.info(response.message.thinking)
                content = response.message.content
                if content is None:
                    raise RuntimeError(
                        f"Failed to get response from agent {self.agent_name}"
                    )
                self.logger.debug(
                    "Content from agent %s:\n%s", self.agent_name, content
                )
                # Append the assistant message because it doesn't know that it
                # actually did anything
                self.message_history.append({"role": "assistant", "content": content})

                stripped_content = self._strip_markdown_formatting(content)
                validated = self.response_model.model_validate_json(stripped_content)
                validated = validated.model_copy(
                    update={
                        "message": self._strip_markdown_formatting(validated.message)
                    }
                )

                self._validate_agent_logic(response=validated)
                if validated.response.type == "final":
                    self._on_response(response=validated)
                    return validated
                if validated.response.type == "tool_call":
                    response_try = 0
                    tool_call = validated.response.tool_call
                    # Check if a tool call is present in a `tool_call` typed response
                    if tool_call:
                        self.logger.debug("Tool call try: %s", tool_try)
                        self._handle_tool_call(tool_call=tool_call, deps=deps)

            except ValidationError as exc:
                error_message = self._print_validation_errors(exc.errors())
                self.message_history.append({"role": "user", "content": error_message})
                self._on_response_retry(
                    response_try=response_try,
                    response_str=content or "None",
                    error_message=error_message,
                )
                response_try += 1
            except AgentExceptionError as exc:
                self.message_history.append({"role": "user", "content": exc.message})
                self._on_response_retry(
                    response_try=response_try,
                    response_str=content or "None",
                    error_message=exc.message,
                )
                response_try += 1
            except ToolExceptionError as exc:
                self.message_history.append({"role": "user", "content": exc.message})
                tool_try += 1
            except (AgentExceptionFatal, ToolExceptionFatal) as exc:
                raise RuntimeError(exc.message)
            except Exception as exc:
                raise RuntimeError(f"Unhandled exception: {str(exc)}")

        if tool_try >= self.tool_retries:
            raise RuntimeError(
                f"Agent {self.agent_name} exhausted tool retries ({self.tool_retries})"
            )
        raise RuntimeError(
            f"Agent {self.agent_name} exhausted response retries ({self.response_retries})"
        )
