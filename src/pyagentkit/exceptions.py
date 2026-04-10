# src/pyagentkit/exceptions.py


class PyAgentKitError(Exception):
    """Base exception for all pyagentkit errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ExceptionAgentError(PyAgentKitError):
    """Recoverable response exception"""

    def __init__(self, message: str):
        super().__init__(message)


class ExceptionAgentFatal(PyAgentKitError):
    """Irrecoverable response exception"""

    def __init__(self, message: str):
        super().__init__(message)


class ExceptionToolError(PyAgentKitError):
    """Recoverable tool calling exception"""

    def __init__(self, message: str):
        super().__init__(message)


class ExceptionToolFatal(PyAgentKitError):
    """Irrecoverable tool exception"""

    def __init__(self, message: str):
        super().__init__(message)


class ExceptionToolRetriesExhausted(PyAgentKitError):
    """Raised when agent has exhausted its tool retries"""

    def __init__(self, agent_name: str, retries: int):
        self.agent_name = agent_name
        self.retries = retries
        super().__init__(
            f"Agent {self.agent_name} has exhaused {self.retries} tool tries"
        )


class ExceptionResponseRetriesExhausted(PyAgentKitError):
    """Raised when agent has exhausted its response retries"""

    def __init__(self, agent_name: str, retries: int):
        self.agent_name = agent_name
        self.retries = retries
        super().__init__(
            f"Agent {self.agent_name} has exhaused {self.retries} response tries"
        )


class ExceptionEnvironmentError(PyAgentKitError):
    """Raised if Ollama server is unreachable or model is not found"""


class ExceptionInvalidTool(PyAgentKitError):
    """Raised when tooling has missing or invalid fields"""


class ExceptionFatalError(PyAgentKitError):
    """Raised when fatal tool or agent error is caught"""


class ExceptionUnhandledError(Exception):
    """Raised for unhandled errors"""
