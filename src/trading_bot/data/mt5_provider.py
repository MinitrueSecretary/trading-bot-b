"""MT5 data provider — Windows only, requires MetaTrader 5 terminal."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trading_bot.config import Settings, Timeframe
from trading_bot.data.base_provider import BaseDataProvider
from trading_bot.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Timeframe mapping from Timeframe enum to MT5 timeframe constants
_MT5_TF_MAP: dict[str, int] = {
    '1m': 1,  # mt5.TIMEFRAME_M1
    '5m': 5,  # mt5.TIMEFRAME_M5
    '15m': 15,  # mt5.TIMEFRAME_M15
    '30m': 30,  # mt5.TIMEFRAME_M30
    '1h': 16385,  # mt5.TIMEFRAME_H1
    '4h': 16388,  # mt5.TIMEFRAME_H4
    '1d': 16408,  # mt5.TIMEFRAME_D1
    '1wk': 32769,  # mt5.TIMEFRAME_W1
}


class MT5Provider(BaseDataProvider):
    """Fetches Forex OHLCV data and account info via MetaTrader 5.

    Only available on Windows with the MetaTrader 5 terminal installed and
    the ``MetaTrader5`` Python package.  On other platforms this class can be
    instantiated but :meth:`connect` will raise :class:`RuntimeError`.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise MT5Provider.

        Args:
            settings: Application :class:`~trading_bot.config.Settings`.
        """
        self._settings = settings
        self._connected: bool = False
        self._mt5: object | None = None

    # ------------------------------------------------------------------
    # BaseDataProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return provider name."""
        return 'MT5'

    def connect(self) -> bool:
        """Initialise the MT5 terminal connection.

        Returns:
            ``True`` on success.

        Raises:
            RuntimeError: On non-Windows systems or if MT5 is not installed.
        """
        if sys.platform != 'win32':
            raise RuntimeError(
                'MT5Provider is only available on Windows. Use YFinanceProvider on other platforms.'
            )

        try:
            import MetaTrader5 as mt5  # type: ignore[import]  # noqa: N813
        except ImportError as exc:
            raise RuntimeError(
                'MetaTrader5 package not installed. Run: uv pip install MetaTrader5'
            ) from exc

        self._mt5 = mt5

        kwargs: dict[str, object] = {}
        if self._settings.mt5_login:
            kwargs['login'] = self._settings.mt5_login
        if self._settings.mt5_password:
            kwargs['password'] = self._settings.mt5_password
        if self._settings.mt5_server:
            kwargs['server'] = self._settings.mt5_server

        if not mt5.initialize(**kwargs):  # type: ignore[attr-defined]
            error = mt5.last_error()  # type: ignore[attr-defined]
            logger.error(f'[MT5] Initialize failed: {error}')
            return False

        account_info = mt5.account_info()  # type: ignore[attr-defined]
        if account_info is None:
            logger.error('[MT5] Could not retrieve account info')
            return False

        logger.info(
            f'[MT5] Connected — login={account_info.login} '
            f'server={account_info.server} '
            f'balance={account_info.balance:.2f} {account_info.currency}'
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 terminal connection."""
        if self._mt5 is not None:
            self._mt5.shutdown()  # type: ignore[union-attr]
            logger.info('[MT5] Disconnected')
        self._connected = False

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        bars: int = 500,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from MT5.

        Args:
            symbol: MT5 symbol string (e.g. ``'EURUSD'``).
            timeframe: Candle timeframe.
            bars: Number of bars to retrieve (ignored when *start* is given).
            start: Inclusive range start in UTC.
            end: Inclusive range end in UTC (defaults to now when *start* given).

        Returns:
            DataFrame with lowercase OHLCV columns and UTC DatetimeIndex.

        Raises:
            RuntimeError: If not connected or MT5 returns no data.
        """
        if not self._connected or self._mt5 is None:
            raise RuntimeError('MT5Provider is not connected. Call connect() first.')

        mt5 = self._mt5
        mt5_tf = _MT5_TF_MAP[timeframe.value]
        mt5_symbol = self._settings.mt5_symbol_for(symbol)

        # Symbol must be selected in Market Watch before history is accessible
        mt5.symbol_select(mt5_symbol, True)  # type: ignore[union-attr]

        if start is not None:
            date_from = (start if start.tzinfo is None else start.replace(tzinfo=None))
            raw_end = end if end is not None else datetime.now(UTC)
            date_to = (raw_end if raw_end.tzinfo is None else raw_end.replace(tzinfo=None))
            rates = self._chunked_rates_range(mt5, mt5_symbol, mt5_tf, date_from, date_to)
        else:
            rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, bars)  # type: ignore[union-attr]

        if rates is None or len(rates) == 0:
            error = mt5.last_error()  # type: ignore[union-attr]
            raise RuntimeError(f'MT5 returned no data for {mt5_symbol}: {error}')

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.rename(
            columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
            },
            inplace=True,
        )
        return df[['open', 'high', 'low', 'close', 'volume']].copy()

    def _chunked_rates_range(
        self,
        mt5: object,
        mt5_symbol: str,
        mt5_tf: int,
        date_from: datetime,
        date_to: datetime,
        chunk_days: int = 90,
    ) -> np.ndarray | None:
        """Fetch rates in chunks to stay within MT5's per-call bar limit.

        Args:
            mt5: Live MetaTrader5 module reference.
            mt5_symbol: Broker symbol name.
            mt5_tf: MT5 timeframe integer constant.
            date_from: Naive UTC start datetime.
            date_to: Naive UTC end datetime.
            chunk_days: Window size in days per request.

        Returns:
            Concatenated structured array, or ``None`` if nothing was returned.
        """
        delta = timedelta(days=chunk_days)
        chunks: list[np.ndarray] = []
        window_start = date_from
        while window_start < date_to:
            window_end = min(window_start + delta, date_to)
            rates = mt5.copy_rates_range(mt5_symbol, mt5_tf, window_start, window_end)  # type: ignore[union-attr]
            if rates is not None and len(rates) > 0:
                chunks.append(rates)
            window_start = window_end + timedelta(seconds=1)
        return np.concatenate(chunks) if chunks else None

    def get_current_price(self, symbol: str) -> float:
        """Return the latest ask price for *symbol* from MT5.

        Args:
            symbol: MT5 symbol string.

        Returns:
            Current ask price.

        Raises:
            RuntimeError: If not connected or symbol info is unavailable.
        """
        if not self._connected or self._mt5 is None:
            raise RuntimeError('MT5Provider is not connected.')

        mt5 = self._mt5
        mt5_symbol = self._settings.mt5_symbol_for(symbol)
        tick = mt5.symbol_info_tick(mt5_symbol)  # type: ignore[union-attr]
        if tick is None:
            raise RuntimeError(f'Cannot get tick for {mt5_symbol}: {mt5.last_error()}')  # type: ignore[union-attr]
        return float(tick.ask)

    # ------------------------------------------------------------------
    # MT5-specific helpers
    # ------------------------------------------------------------------

    def get_account_balance(self) -> float:
        """Return current account balance.

        Returns:
            Account balance in account currency.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._mt5 is None:
            raise RuntimeError('MT5Provider is not connected.')
        info = self._mt5.account_info()  # type: ignore[union-attr]
        return float(info.balance)

    def get_open_positions(self) -> pd.DataFrame:
        """Return all open positions as a DataFrame.

        Returns:
            DataFrame of open positions (empty if none).
        """
        if not self._connected or self._mt5 is None:
            raise RuntimeError('MT5Provider is not connected.')
        positions = self._mt5.positions_get()  # type: ignore[union-attr]
        if not positions:
            return pd.DataFrame()
        return pd.DataFrame([p._asdict() for p in positions])
