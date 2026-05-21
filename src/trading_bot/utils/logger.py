"""Logging utilities using Rich for enhanced terminal output."""

import logging

from rich.console import Console
from rich.logging import RichHandler

_console = Console()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger with Rich formatting.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(console=_console, rich_tracebacks=True, show_path=False)],
    )
    return logging.getLogger(name)


console = _console
