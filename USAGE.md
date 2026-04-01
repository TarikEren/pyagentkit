# How To Use Pyagentkit

## Defining Agents

- Defining a `base` agent is as easy as:

```python

from pyagentkit import Agent

agent = Agent(llm_name="<llm_name>")
response = agent.handle_response(prompt="Hello, World!")

```

where `llm_name` is the name of a large language model downloaded locally with Ollama

- Custom Pydantic return schemas can also be defined for agents:
```python

from pyagentkit import Agent
from pyagentkit.definitions import AgentResponse

# Custom output type
class CustomOutput(AgentResponse):
    sum_result: int

# Agent definition
agent = Agent(llm_name="<llm_name>", response_model=CustomOutput)

# Response handler
response = agent.handle_response(prompt="What is 2 + 2")

# The custom output field can be accessed now.
print(f"Result: {response.sum_result}")

```

## Defining Tools

- Tools are python functions that:
    - are registered to an agent with decorator `register_tool`:
    - are described with a concise docstring
    - return a `ToolResult` object

That's it.

### Defining Basic Tools

```python
from pyagentkit import Agent
from pyagentkit.definitions import AgentResponse

# Custom output type
class CustomOutput(AgentResponse):
    sum_result: int

# Agent definition
agent = Agent(llm_name="<llm_name>", response_model=CustomOutput)

# Custom tool with registration
@agent.register_tool
def add_tool(n1: int, n2: int):
    """Adds two numbers""" # Short docstring

    # Generic return type for ALL tooling
    return ToolResult(
        return_value=ToolReturnValue.success,   # Denotes the tool result 
        content=f"Result: {n1 + n2}"            # This part will be `fed into` the agent to use
    )

# Response handler
response = agent.handle_response(prompt="What is 2 + 2")

# The custom output field can be accessed just like the other example.
print(f"Result: {response.sum_result}")
```

#### Tool Return Values
| Value                       | Behaviour                                                                   |
|-----------------------------|-----------------------------------------------------------------------------|
| `tool_return_value.success` | Tool succeeded. `content` is fed back to the agent.                         |
| `tool_return_value.error`   | Recoverable failure. `content` is fed back to the agent to retry or adjust. |
| `tool_return_value.fatal`   | Unrecoverable failure. Execution stops immediately.                         |

### Defining Tools With Dependencies


- Dependencies are classes that inherit the class `AgentDependencies`
    - NOTE: `AgentDependencies` has a pre-defined `prompt` field in it that should be filled when
    using.

- When defining tools with dependencies, the dependency must ALWAYS be the FIRST parameter of the tool.

```python
from pyagentkit import Agent
from pyagentkit.definitions import (
    AgentDependencies,
    AgentResponse,
    ToolResult,
    ToolReturnValue,
)
# Custom output class
class CustomOutput(AgentResponse):
    data: object


# Agent definition
agent = Agent(llm_name="<llm_name>", response_model=CustomOutput)

# Custom dependencies to inject into tooling
# Class AgentDependencies has a prompt field in it
class CustomDeps(AgentDependencies):
    db_url: str
    api_key: str

deps = MyDeps(prompt="Fetch John Doe from db", db_url="...", api_key="...")

# Custom database access tool
@agent.register_tool
def fetch_from_db(deps: CustomDeps, name: str):
    """Fetches details of a person from db"""

    # db_url and api_key now can be accessed from the tool
    db = connect_to_db(url=deps.db_url, key=deps.api_key)

    return ToolResult(
        return_value=ToolReturnValue.success,
        content=f"Data for `{name}` in db: {db.fetch(name)}",
    )


def main():
    # Dependencies object for the agent to use
    deps = MyDeps(prompt="Fetch John Doe from db", db_url="...", api_key="...")

    # Response handler
    result = agent.handle_response(
        prompt=deps.prompt, # The prompt from the dependencies can be used
        dependencies=deps   # The dependency object
    )

    # The data field can be accessed just like before
    print(f"Data of `John Doe`: {result.data}")

```

The only restriction when it comes to define dependencies is that they should inherit class `AgentDependencies`

