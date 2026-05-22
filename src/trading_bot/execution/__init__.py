"""Execution package — platform-aware factory."""

import sys

from trading_bot.config import Settings
from trading_bot.execution.base_executor import BaseExecutor, OrderResult, OrderStatus
from trading_bot.execution.paper_executor import PaperExecutor


def get_executor(settings: Settings) -> BaseExecutor:
    """Return the appropriate executor for the current platform and config.

    - If ``settings.dry_run`` is ``True`` (default): always returns :class:`PaperExecutor`.
    - On **Windows** with ``dry_run=False``: returns :class:`MT5Executor`.
    - On non-Windows with ``dry_run=False``: logs a warning and falls back to paper.

    Args:
        settings: Application settings.

    Returns:
        A concrete :class:`BaseExecutor`.
    """
    from trading_bot.utils.logger import get_logger

    logger = get_logger(__name__)

    if settings.dry_run:
        logger.info('[Executor] dry_run=True — using PaperExecutor')
        return PaperExecutor()

    if sys.platform == 'win32':
        from trading_bot.execution.mt5_executor import MT5Executor

        logger.info('[Executor] Windows detected + dry_run=False — using MT5Executor')
        return MT5Executor(settings)

    logger.warning(
        '[Executor] Non-Windows platform with dry_run=False; '
        'falling back to PaperExecutor (MT5 unavailable)'
    )
    return PaperExecutor()


__all__ = ['BaseExecutor', 'OrderResult', 'OrderStatus', 'PaperExecutor', 'get_executor']
