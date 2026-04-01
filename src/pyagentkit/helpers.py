import logging


def configure_logging(level: int = logging.INFO) -> None:
    """
    Enables pyagentkit log output.
    Call this at the start of your script to see agent messages.

    Args:
        level: Logging level (default: logging.INFO)
    """
    logger = logging.getLogger("pyagentkit")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
