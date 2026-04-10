# PyAgentKit

A Python library for building tool-calling agents on top of locally-hosted LLMs via [Ollama](https://ollama.com). PyAgentKit gives models that lack native function-calling support the ability to call tools through structured JSON output, retries, dependency injection, and lifecycle hooks.

---

## Features

- **Sync and async agents** — `Agent` for synchronous use, `AsyncAgent` for `asyncio`-based workflows
- **Tool registration** — attach tools at the class level (shared across all instances) or at the instance level (per-agent)
- **Structured JSON responses** — agents respond in a validated, discriminated-union schema (`final` or `tool_call`)
- **Pydantic response models** — extend `AgentResponse` to add your own typed fields to every response
- **Dependency injection** — pass runtime dependencies (database connections, config, etc.) through `AgentDependencies` into tools without polluting tool signatures
- **Retry logic** — configurable retry budgets separately for tool calls and for response validation failures
- **Approval gates** — optionally require human confirmation before any tool is executed
- **Lifecycle hooks** — callbacks for tool calls, retries, successes, and final responses
- **Agent composition** — expose any agent as a tool that another agent can call via `.as_tool()`
- **Token usage tracking** — cumulative `TokenUsage` object updated after every LLM call
- **Message history** — persistent within a session, with optional trimming, save, and load
- **Thinking support** — pass `think=True` to enable chain-of-thought reasoning on supported models

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally (or at a reachable URL)
- A model pulled in Ollama (e.g. `ollama pull llama3.2`)

---

## Installation

```bash
pip install pyagentkit
```

---

## Project Structure

```
src/pyagentkit/
├── agent.py          # Synchronous Agent class
├── async_agent.py    # Asynchronous AsyncAgent class
├── definitions.py    # Pydantic models, type aliases, enums
└── exceptions.py     # Exception hierarchy
```

---

## Core Concepts

### Response Schema

Every agent responds in one of two JSON shapes, validated via Pydantic:

```json
// Tool call
{
  "response": {
    "type": "tool_call",
    "tool_call": { "name": "my_tool", "params": [{ "name": "x", "value": "42" }] }
  },
  "message": "Calling my_tool to get the result"
}

// Final answer
{
  "response": { "type": "final" },
  "message": "The answer is 42"
}
```

### Exception Hierarchy

```
PyAgentKitError
├── ExceptionAgentError          # Recoverable response failure (triggers retry)
├── ExceptionAgentFatal          # Irrecoverable response failure
├── ExceptionToolError           # Recoverable tool failure (triggers retry)
├── ExceptionToolFatal           # Irrecoverable tool failure
├── ExceptionToolRetriesExhausted
├── ExceptionResponseRetriesExhausted
├── ExceptionEnvironmentError    # Ollama unreachable or model not found
├── ExceptionInvalidTool         # Tool missing docstring or malformed
└── ExceptionFatalError          # Wraps any fatal agent or tool exception
ExceptionUnhandledError          # Unhandled runtime exception (not a PyAgentKitError)
```

---

## License

APACHE 2.0
