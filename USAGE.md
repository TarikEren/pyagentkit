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


# Base agent definition
agent = Agent(llm_name="<llm_name>")

# Custom output type
class CustomOutput(AgentResponse):
    sum_result: int

# Base agent definition with custom output
agent_with_custom_output = Agent(llm_name="<llm_name>", response_model=CustomOutput)

# Response handler
response = agent_with_custom_output.handle_response(prompt="What is 2 + 2")

# The custom output field can be accessed now.
print(f"Result: {response.sum_result}")

```

- Custom agent classes can be used to add more customization:
```python
from pyagentkit import Agent
from pyagentkit.definitions import AgentResponse

# Custom output type
class CustomOutput(AgentResponse):
    response_value: int

# Custom agent class for more customization
class CustomAgentClass(Agent):
    field1: int
    field2: str
    ...

# Custom agent definition
customized_agent = CustomAgentClass(llm_name="", response_model=CustomOutput)

# Response handler works just the same
response = customized_agent.handle_response(prompt="Some kind of task")

# So does fetching fields
print(f"Result: {response.response_value}")
```

## Defining Tools

- Tools are python functions that:
    - are registered to an agent with decorator `register_tool`:
    - are described with a concise docstring
    - return a `ToolResult` object

That's it.

- Tools can be defined class-wide using the `register_tool` decorator or implementation-wide using `add_tool` function

### Defining Basic Tools

#### Class-Wide Definition
```python
from pyagentkit import Agent
from pyagentkit.definitions import AgentResponse

# Custom output type
class CustomOutput(AgentResponse):
    result: int

class CustomAgentClass(Agent):
    ...

# Custom agent definition
agent1 = CustomAgentClass(llm_name="<llm_name>", response_model=CustomOutput)

# Class-wide tool registration
# Every agent that uses class `CustomAgentClass` will now
# have access to the following tool
@Agent.register_tool
def addition(n1: int, n2: int):
    """Adds two numbers""" # Short docstring

    # Generic return type for ALL tooling
    return ToolResult(
        return_value=ToolReturnValue.success,   # Denotes the tool result 
        content=f"Result: {n1 + n2}"            # This part will be `fed into` the agent to use
    )

# This agent ALSO has access to the `addition` tool
# due to class-wide registration
agent2 = CustomAgentClass(llm_name="<llm_name>", response_model=CustomOutput)
```

#### Instance Wide-Definition

```python
from pyagentkit import Agent
from pyagentkit.definitions import AgentResponse

# Custom output type
class CustomOutput(AgentResponse):
    result: int

class CustomAgentClass(Agent):
    ...

# Custom agent definition
agent1 = CustomAgentClass(llm_name="<llm_name>", response_model=CustomOutput)
agent2 = CustomAgentClass(llm_name="<llm_name>", response_model=CustomOutput)


# Tool definition (Notice the absence of the decorator `register_tool`)
def subtraction(n1: int, n2: int):
    """Subtracts one number from another"""

    return ToolResult(
        return_value=ToolReturnValue.success,
        content=f"Result: {n1 - n2}"
    )

# Instance based tool registration
# Only agent1 has access to the following tool
agent1.add_tool(subtraction)
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

deps = CustomDeps(prompt="Fetch John Doe from db", db_url="...", api_key="...")

# Custom database access tool
def fetch_from_db(deps: CustomDeps, name: str):
    """Fetches details of a person from db"""

    # db_url and api_key now can be accessed from the tool
    db = connect_to_db(url=deps.db_url, key=deps.api_key)

    return ToolResult(
        return_value=ToolReturnValue.success,
        content=f"Data for `{name}` in db: {db.fetch(name)}",
    )

# Instance-wide registration
agent.add_tool(fetch_from_db)


def main():
    # Dependencies object for the agent to use
    deps = CustomDeps(prompt="Fetch John Doe from db", db_url="...", api_key="...")

    # Response handler
    result = agent.handle_response(
        prompt=deps.prompt, # The prompt from the dependencies can be used
        deps=deps   # The dependency object
    )

    # The data field can be accessed just like before
    print(f"Data of `John Doe`: {result.data}")

```

The only restriction when it comes to define dependencies is that they should inherit class `AgentDependencies`

