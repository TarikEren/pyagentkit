class AgentExceptionError(Exception):
    """Recoverable response exception"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class AgentExceptionFatal(Exception):
    """Irrecoverable response exception"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ToolExceptionError(Exception):
    """Recoverable tool calling exception"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ToolExceptionFatal(Exception):
    """Irrecoverable tool exception"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
