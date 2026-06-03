"""YFinance data provider — cross-platform, no broker account needed."""

from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from trading_bot.config import Timeframe
from trading_bot.data.base_provider import BaseDataProvider
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from Timeframe enum to yfinance interval string
_TF_MAP: dict[Timeframe, str] = {
    Timeframe.M1: '1m',
    Timeframe.M5: '5m',
    Timeframe.M15: '15m',
    Timeframe.M30: '30m',
    Timeframe.H1: '1h',
    Timeframe.H4: '1h',  # yfinance has no 4h; caller should aggregate
    Timeframe.D1: '1d',
    Timeframe.W1: '1wk',
}

# yfinance requires a `period` parameter when no explicit start/end is given.
# We map requested bar counts to the smallest sufficient period.
_PERIOD_FOR_TF: dict[Timeframe, str] = {
    Timeframe.M1: '7d',
    Timeframe.M5: '60d',
    Timeframe.M15: '60d',
    Timeframe.M30: '60d',
    Timeframe.H1: '730d',
    Timeframe.H4: '730d',
    Timeframe.D1: 'max',
    Timeframe.W1: 'max',
}


def _to_yf_symbol(symbol: str) -> str:
    """Convert MT5-style symbol (``EURUSD``) to yfinance format (``EURUSD=X``)."""
    if symbol.endswith('=X'):
        return symbol
    return f'{symbol}=X'


class YFinanceProvider(BaseDataProvider):
    """Fetches Forex OHLCV data via the ``yfinance`` library.

    This provider is intended for development / backtesting on non-Windows
    systems.  Live prices come from Yahoo Finance and may lag by several
    minutes; do **not** use for real-time execution.
    """

    def __init__(self) -> None:
        self._connected: bool = False

    # ------------------------------------------------------------------
    # BaseDataProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return provider name."""
        return 'YFinance'

    def connect(self) -> bool:
        """YFinance requires no persistent connection — always succeeds."""
        self._connected = True
        logger.info('[YFinance] Provider ready (no persistent connection required)')
        return True

    def disconnect(self) -> None:
        """No teardown required for YFinance."""
        self._connected = False
        logger.info('[YFinance] Disconnected')

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        bars: int = 500,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: MT5-style symbol (e.g. ``'EURUSD'``) or YF-style (``'EURUSD=X'``).
            timeframe: Candle timeframe.
            bars: Approximate number of bars to return (ignored when *start* is given).
            start: Inclusive range start in UTC.
            end: Inclusive range end in UTC (defaults to now when *start* given).

        Returns:
            DataFrame with lowercase OHLCV columns and UTC DatetimeIndex.

        Raises:
            ValueError: If no data is returned for the symbol.
        """
        yf_symbol = _to_yf_symbol(symbol)
        interval = _TF_MAP[timeframe]

        ticker = yf.Ticker(yf_symbol)

        if start is not None:
            date_from = start if start.tzinfo is not None else start.replace(tzinfo=UTC)
            raw_end = end if end is not None else datetime.now(UTC)
            date_to = raw_end if raw_end.tzinfo is not None else raw_end.replace(tzinfo=UTC)
            logger.info(
                f'[YFinance] Downloading {yf_symbol} | tf={interval} '
                f'| {date_from.date()} → {date_to.date()}'
            )
            df: pd.DataFrame = ticker.history(
                start=date_from, end=date_to, interval=interval, auto_adjust=True
            )
        else:
            period = _PERIOD_FOR_TF[timeframe]
            logger.info(f'[YFinance] Downloading {yf_symbol} | tf={interval} | period={period}')
            df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            raise ValueError(f'No data returned for symbol {yf_symbol!r}')

        # Normalise column names to lowercase
        df.columns = df.columns.str.lower()

        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Ensure UTC timezone-aware index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        # Remove any NaN rows
        df.dropna(inplace=True)

        # Return the most recent `bars` rows
        return df.iloc[-bars:] if len(df) > bars else df

    def get_current_price(self, symbol: str) -> float:
        """Return the latest closing price for *symbol*.

        Args:
            symbol: MT5-style or YF-style symbol.

        Returns:
            Latest closing price.

        Raises:
            ValueError: If price cannot be retrieved.
        """
        yf_symbol = _to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        info = ticker.fast_info
        price: float = float(info.last_price)
        if not price:
            raise ValueError(f'Unable to retrieve current price for {yf_symbol!r}')
        return price
