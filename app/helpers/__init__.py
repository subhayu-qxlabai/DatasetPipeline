from functools import partial

from .utils import *
from .types import *
from .logger import make_logger, LOGGER

run_parallel_exec = partial(run_parallel_exec, error_logger=LOGGER.error)
