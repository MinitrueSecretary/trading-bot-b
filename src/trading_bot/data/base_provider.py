"""Abstract base class for OHLCV data providers."""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from trading_bot.config import Timeframe


class BaseDataProvider(ABC):
    """Interface that every data provider must implement.

    Concrete implementations: :class:`~trading_bot.data.yfinance_provider.YFinanceProvider`
    (cross-platform) and :class:`~trading_bot.data.mt5_provider.MT5Provider` (Windows).
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection / initialise the data source.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection / release resources."""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        bars: int = 500,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars.

        Supply either *bars* (most-recent N candles) or *start* / *end* UTC
        datetimes (date-range fetch).  When *start* and/or *end* are given
        they take precedence and *bars* is ignored.

        Args:
            symbol: Instrument symbol in the provider's native format.
            timeframe: Candle timeframe.
            bars: Number of historical bars to fetch (used when no date range given).
            start: Inclusive range start in UTC.  If only *start* is given,
                *end* defaults to the current UTC time.
            end: Inclusive range end in UTC.  If only *end* is given,
                *start* must also be provided.

        Returns:
            DataFrame with columns ``['open', 'high', 'low', 'close', 'volume']``
            and a timezone-aware DatetimeIndex (UTC).
        """

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Return the latest mid/ask price for *symbol*.

        Args:
            symbol: Instrument symbol.

        Returns:
            Current price as a float.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. ``'YFinance'``)."""
