## PyAgentKit

***NOTE: Currently work in progress, functionality may not work as intended***

Lightweight library for creating tool-enabled AI agents using Ollama. Including the ones that do not have tool calling capabilities.

## Why this exists

- To give models which cannot produce structured tool calls a way to call custom made Python tooling.

## Features

- Works with models that have no native tool calling
- Easy, simple tool creation and robust execution loop
- Schema validation for tool inputs
- Structured and validated agent responses

## Installation (Not in PyPI yet)

```bash
pip install pyagentkit
```

## Example

```python
from ollama_agentkit import Agent, tool, AgentResponse

# Custom output class to manipulate LLM output
class CustomOutput(AgentResponse):
    sum: int

agent = Agent(
    # Works with any LLM's locally downloaded in your ollama server
    llm_name="qwen2.5-coder:7b",

    # Custom response models are allowed for responses
    response_model=CustomOutput
)

@agent.register_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Messages get printed during the response handling
output = agent.handle_response("What is 12 + 30?")

# You can access the CustomOutput value afterwards
print(output.response.sum)

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
