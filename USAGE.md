# PyAgentKit — Usage Guide

---

## Table of Contents

1. [Basic Agent](#1-basic-agent)
2. [Writing Tools](#2-writing-tools)
3. [Instance vs Class Tools](#3-instance-vs-class-tools)
4. [Dependencies](#4-dependencies)
5. [Custom Response Models](#5-custom-response-models)
6. [Async Agent](#6-async-agent)
7. [Lifecycle Hooks](#7-lifecycle-hooks)
8. [Agent Composition](#8-agent-composition)
9. [Message History](#9-message-history)
10. [Token Usage](#10-token-usage)
11. [Error Handling](#11-error-handling)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. Basic Agent

```python
from pyagentkit import Agent

agent = Agent(
    llm_name="llama3.2",          # Must be pulled in Ollama
    system_prompt="You are a helpful assistant.",
    agent_name="my-agent",
)

response = agent.handle_response("What is 2 + 2?")
print(response.message)           # "The answer is 4"
```

`handle_response` blocks until the agent produces a `final` response or exhausts its retries.

---

## 2. Writing Tools

A tool is any plain Python function that returns a `ToolResult`. It **must** have a docstring — the docstring is what the agent reads to understand what the tool does.

```python
from pyagentkit import Agent
from pyagentkit.definitions import ToolResult, ToolReturnValue


def add_numbers(a: int, b: int) -> ToolResult:
    """Add two integers together and return their sum."""
    total = a + b
    return ToolResult(return_value=ToolReturnValue.success, content=str(total))


def divide(a: float, b: float) -> ToolResult:
    """Divide a by b. Returns an error if b is zero."""
    if b == 0:
        return ToolResult(return_value=ToolReturnValue.error, content="Cannot divide by zero")
    return ToolResult(return_value=ToolReturnValue.success, content=str(a / b))
```

### ToolReturnValue options

| Value | Effect |
|-------|--------|
| `success` | Result appended to history; agent continues |
| `error` | Error message sent back to agent; counts as a tool retry |
| `fatal` | Immediately raises `ExceptionFatalError`; no retry |

### Passing tools to an agent

```python
agent = Agent(
    llm_name="llama3.2",
    tools=[add_numbers, divide],  # registered without approval requirement
)
```

By default, tools added via the `tools=` constructor parameter require approval (`requires_approval=True`). To skip the prompt, use `add_tool` with `requires_approval=False`:

```python
agent.add_tool(add_numbers, requires_approval=False)
```

---

## 3. Instance vs Class Tools

### Instance tools

Registered on a single agent instance. Use `add_tool` or pass `tools=` to the constructor.

```python
agent = Agent(llm_name="llama3.2")
agent.add_tool(my_tool, requires_approval=False)
```

### Class tools

Registered on the class and shared across **all instances** of that subclass.

```python
class MyAgent(Agent):
    pass


@MyAgent.register_tool(requires_approval=False)
def get_time() -> ToolResult:
    """Return the current UTC time as an ISO string."""
    from datetime import datetime, timezone
    return ToolResult(
        return_value=ToolReturnValue.success,
        content=datetime.now(timezone.utc).isoformat(),
    )


agent_a = MyAgent(llm_name="llama3.2")
agent_b = MyAgent(llm_name="llama3.2", agent_name="second")
# Both agents can call get_time
```

---

## 4. Dependencies

Use `AgentDependencies` to inject runtime state (database handles, API clients, config) into tools without exposing them in the tool's public signature shown to the LLM.

```python
from pydantic import Field
from pyagentkit.definitions import AgentDependencies, ToolResult, ToolReturnValue


class MyDeps(AgentDependencies):
    db_url: str = Field(description="Database connection URL")
    api_key: str = Field(description="External API key")


def fetch_user(deps: MyDeps, user_id: str) -> ToolResult:
    """Fetch a user record from the database by user ID."""
    # deps.db_url and deps.api_key are available here
    # but `deps` is hidden from the LLM's tool signature
    result = f"User {user_id} from {deps.db_url}"
    return ToolResult(return_value=ToolReturnValue.success, content=result)


deps = MyDeps(prompt="Find user 42", db_url="postgresql://...", api_key="sk-...")

agent = Agent(llm_name="llama3.2", tools=[fetch_user])
response = agent.handle_response("Find user 42", deps=deps)
```

The `deps` first parameter is detected automatically by inspecting whether the first parameter is a subclass of `AgentDependencies`. It is stripped from the tool signature shown to the LLM.

---

## 5. Custom Response Models

Extend `AgentResponse` to add typed fields that the agent must populate on every response.

```python
from pydantic import Field
from pyagentkit.definitions import AgentResponse


class SentimentResponse(AgentResponse):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")


agent = Agent(
    llm_name="llama3.2",
    response_model=SentimentResponse,
    system_prompt="You are a sentiment analysis assistant.",
)

response = agent.handle_response("I love this product!")
print(response.sentiment)    # "positive"
print(response.confidence)   # 0.95
```

The schema examples injected into the system prompt are generated automatically from the model's field annotations, so the LLM always sees the correct shape.

---

## 6. Async Agent

Use `AsyncAgent` in `asyncio` contexts. The API mirrors `Agent` except all relevant methods are coroutines and the client is `ollama.AsyncClient`.

```python
import asyncio
from pyagentkit import AsyncAgent
from pyagentkit.definitions import ToolResult, ToolReturnValue


async def get_weather(city: str) -> ToolResult:
    """Return current weather for a given city name."""
    # ... call a weather API ...
    return ToolResult(return_value=ToolReturnValue.success, content=f"Sunny in {city}")


async def main():
    agent = await AsyncAgent.create(
        llm_name="llama3.2",
        tools=[get_weather],
        agent_name="weather-agent",
    )
    response = await agent.handle_response("What is the weather in Istanbul?")
    print(response.message)
    agent.dispose()


asyncio.run(main())
```

> **Important:** Use `await AsyncAgent.create(...)` instead of `AsyncAgent(...)` directly. The constructor is synchronous but environment verification (`_verify_ollama_environment`) is async and must be awaited via `create`.

---

## 7. Lifecycle Hooks

Hooks let you observe and log what the agent is doing without modifying its core logic.

```python
from pyagentkit.definitions import AgentResponse


def on_tool_call(tool_name: str, params: dict) -> None:
    print(f"[CALL]    {tool_name} | params={params}")


def on_tool_retry(tool_name: str, params: dict, error: str) -> None:
    print(f"[RETRY]   {tool_name} | error={error}")


def on_tool_success(tool_name: str, params: dict) -> None:
    print(f"[SUCCESS] {tool_name}")


def on_response(response: AgentResponse) -> None:
    print(f"[FINAL]   {response.message}")


def on_response_retry(attempt: int, response_str: str, error: str) -> None:
    print(f"[RESP RETRY] attempt={attempt} | error={error}")


agent = Agent(
    llm_name="llama3.2",
    on_tool_call=on_tool_call,
    on_tool_retry=on_tool_retry,
    on_tool_success=on_tool_success,
    on_response=on_response,
    on_response_retry=on_response_retry,
)
```

---

## 8. Agent Composition

Any agent can expose itself as a tool for another agent via `.as_tool()`.

```python
from pyagentkit import Agent
from pyagentkit.definitions import ToolResult, ToolReturnValue


# Inner specialist agent
researcher = Agent(
    llm_name="llama3.2",
    agent_name="researcher",
    system_prompt="You are a research specialist. Answer factual questions concisely.",
)

# Outer orchestrator agent
orchestrator = Agent(
    llm_name="llama3.2",
    agent_name="orchestrator",
    system_prompt="You are an orchestrator. Delegate research tasks to the researcher agent.",
    tools=[researcher.as_tool(description="Ask the researcher agent a factual question.")],
)

response = orchestrator.handle_response("What is the capital of Japan?")
print(response.message)
```

`.as_tool()` wraps `handle_response` in a `ToolResult`-returning function and names it after the agent, so the outer agent can discover and call it like any other tool.

---

## 9. Message History

History is maintained automatically within a session. Each call to `handle_response` appends to the same history, giving the agent memory of prior turns.

```python
agent = Agent(llm_name="llama3.2", agent_name="chat-agent")

r1 = agent.handle_response("My name is Alice.")
r2 = agent.handle_response("What is my name?")
print(r2.message)  # "Your name is Alice."
```

### Trimming history

Set `max_history` to limit how many non-system messages the agent sees. Older messages are dropped automatically. Orphaned assistant messages (those without a preceding user message after trimming) are also removed.

```python
agent = Agent(llm_name="llama3.2", max_history=20)
```

### Saving and loading history

```python
agent.save_history("history.json")

# Later, in a new session:
agent.load_history("history.json")
```

### Clearing history

```python
agent.clear_history()
```

---

## 10. Token Usage

`agent.token_usage` is a `TokenUsage` object that accumulates across all `handle_response` calls on an instance.

```python
response = agent.handle_response("Summarize the water cycle.")
print(agent.token_usage.prompt_tokens)
print(agent.token_usage.response_tokens)
print(agent.token_usage.total_tokens)
```

---

## 11. Error Handling

```python
from pyagentkit.exceptions import (
    ExceptionToolRetriesExhausted,
    ExceptionResponseRetriesExhausted,
    ExceptionFatalError,
    ExceptionEnvironmentError,
    ExceptionUnhandledError,
)

try:
    response = agent.handle_response("Do something complex.")
except ExceptionEnvironmentError as e:
    print(f"Ollama not reachable or model missing: {e.message}")
except ExceptionToolRetriesExhausted as e:
    print(f"Tool retries exhausted after {e.retries} attempts")
except ExceptionResponseRetriesExhausted as e:
    print(f"Response retries exhausted after {e.retries} attempts")
except ExceptionFatalError as e:
    print(f"Fatal error, cannot recover: {e.message}")
except ExceptionUnhandledError as e:
    print(f"Unexpected error: {e}")
```

Inside a tool, signal errors to the agent using return values rather than raising exceptions directly:

```python
def risky_tool(path: str) -> ToolResult:
    """Read a file from the given path."""
    try:
        with open(path) as f:
            return ToolResult(return_value=ToolReturnValue.success, content=f.read())
    except FileNotFoundError:
        # Recoverable — agent will retry with corrected params
        return ToolResult(return_value=ToolReturnValue.error, content=f"File not found: {path}")
    except PermissionError:
        # Irrecoverable — raises ExceptionFatalError immediately
        return ToolResult(return_value=ToolReturnValue.fatal, content=f"Permission denied: {path}")
```

---

## 12. Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_name` | `str` | required | Ollama model name (e.g. `"llama3.2"`) |
| `agent_name` | `str \| None` | `llm_name` | Unique name for this agent instance |
| `system_prompt` | `str \| None` | `""` | Base system prompt prepended to all requests |
| `instructions` | `str \| None` | `""` | Text appended to every user prompt |
| `response_model` | `Type[AgentResponse]` | `AgentResponse` | Pydantic model for response validation |
| `tool_retries` | `int` | `3` | Max tool call retries before raising |
| `response_retries` | `int` | `3` | Max response validation retries before raising |
| `num_ctx` | `int` | `8192` | Ollama context window size |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `top_p` | `float \| None` | `None` | Nucleus sampling probability |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `ollama_url` | `str \| None` | `None` | Custom Ollama host URL; uses default if `None` |
| `tools` | `list \| None` | `None` | Tool functions to register on init |
| `max_history` | `int \| None` | `None` | Max non-system messages to retain |
| `log_level` | `int` | `logging.INFO` | Python logging level |
| `think` | `bool` | `False` | Enable chain-of-thought on supported models |
| `on_tool_call` | callable | `None` | Hook called before each tool execution |
| `on_tool_retry` | callable | `None` | Hook called on tool error/retry |
| `on_tool_success` | callable | `None` | Hook called on tool success |
| `on_response` | callable | `None` | Hook called on final response |
| `on_response_retry` | callable | `None` | Hook called on response validation failure |

### Agent lifecycle

```python
agent = Agent(llm_name="llama3.2", agent_name="my-agent")

# Use the agent ...

# When done, unregister and clean up the logger:
agent.dispose()
```

Each agent name must be unique within the `Agent` or `AsyncAgent` registry. Calling `dispose()` frees the name so it can be reused.
