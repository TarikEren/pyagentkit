## PyAgentKit

Lightweight library for creating tool-enabled AI agents using Ollama. Including the ones that do not have tool calling capabilities.

- Refer to [USAGE.md](./USAGE.md) for usage details.

## Why this exists

- To give models which cannot produce structured tool calls a way to call custom made Python tooling.

## Features

- Works with models that have no native tool calling
- Easy, simple tool creation and robust execution loop
- Schema validation for tool inputs
- Structured and validated agent responses

## Installation

```bash
pip install pyagentkit
```

## Example

```python
from ollama_agentkit import Agent, tool, AgentResponse, configure_logging

# For configuring agent logging
configure_logging()

# Custom output class to manipulate LLM output
class CustomOutput(AgentResponse):
    sum_result: int

agent = Agent(
    # Works with any LLM's locally downloaded in your ollama server
    llm_name="qwen2.5-coder:7b",

    # Custom response models are allowed for responses
    response_model=CustomOutput
)

def sum_tool(n1: int, n2: int):
    """Adds two numbers""" # All tooling must have a one-liner docstring that defines what the function does
    # Every tool must return a `ToolResult` object
    return ToolResult(
        return_value=ToolReturnValue.success,     # Status of the return | Can be either success, error or fatal.
                                                  # Affects how the tool call is processed.
        content=f"Result: {n1 + n2}"              # The part which the agent will read
    )

agent.add_tool(sum_tool)

# Messages get printed during the response handling
result = agent.handle_response(prompt="What is 2 + 2")

# Custom output class fields can be accessed as such
print(result.sum_result)

```

## Design goals
- Simple to use
- Model-agnostic where possible
- Strict validation
- Clear separation of agent logic and application logic

## Limitations
- Tool reliability depends on prompt and model instruction-following quality
- Some models may need stronger prompting
- Long tool chains may need retry logic or guardrails
