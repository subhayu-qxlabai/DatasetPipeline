import sys
from typing import Literal
from loguru import logger


def make_logger(
    filename: str = None,
    min_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    enable_backtrace: bool = True,
):
    
    """
    Creates a logger with the specified configuration.
    Args:
        filename (str, optional): The name of the log file. Defaults to None.
        min_level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], optional): The minimum log level. Defaults to "INFO".
        enable_backtrace (bool, optional): Whether to enable backtrace. Defaults to True.
    Returns:
        logger: The logger object.
    Example:
        logger = make_logger(filename="my_log", min_level="DEBUG", enable_backtrace=False)
    """

    if filename is None:
        out = sys.stdout
    else:
        out = f"logs/{filename}.log"
    logger.add(
        sink=out,
        level=min_level,
        colorize=True,
        backtrace=enable_backtrace,
        format="[<green>{time}</green> | {level} | {module} | <level>{message}</level>]",
    )
    return logger


LOGGER = make_logger()
