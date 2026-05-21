"""Data provider package — platform-aware factory function."""

import sys

from trading_bot.config import Settings
from trading_bot.data.base_provider import BaseDataProvider
from trading_bot.data.yfinance_provider import YFinanceProvider


def get_data_provider(settings: Settings) -> BaseDataProvider:
    """Return the appropriate data provider for the current platform.

    On **Windows** the MT5Provider is returned; on all other platforms the
    YFinanceProvider is used.

    Args:
        settings: Application settings.

    Returns:
        A concrete :class:`BaseDataProvider` implementation.
    """
    if sys.platform == 'win32':
        from trading_bot.data.mt5_provider import MT5Provider
        return MT5Provider(settings)
    return YFinanceProvider()


__all__ = ['BaseDataProvider', 'YFinanceProvider', 'get_data_provider']
