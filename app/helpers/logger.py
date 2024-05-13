import sys
from typing import Literal
from loguru import logger


def make_logger(
    filename: str = None,
    min_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    enable_backtrace: bool = True,
):
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
