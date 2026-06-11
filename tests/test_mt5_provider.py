"""Unit tests for MT5Provider using a mocked MetaTrader5 module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from trading_bot.config import Settings, Timeframe
from trading_bot.data.mt5_provider import MT5Provider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rates(n: int = 10) -> np.ndarray:
    """Return a minimal structured array mimicking MT5 rates output."""
    base_ts = 1_735_689_600  # 2025-01-01 00:00:00 UTC
    dtype = [
        ('time', 'i8'),
        ('open', 'f8'),
        ('high', 'f8'),
        ('low', 'f8'),
        ('close', 'f8'),
        ('tick_volume', 'i8'),
        ('spread', 'i4'),
        ('real_volume', 'i8'),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr['time'] = np.arange(base_ts, base_ts + n * 300, 300)
    arr['open'] = 1900.0
    arr['high'] = 1901.0
    arr['low'] = 1899.0
    arr['close'] = 1900.5
    arr['tick_volume'] = 1000
    return arr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> Settings:
    """Settings with XAUUSD and default suffix 'm'."""
    return Settings(
        dry_run=True,
        symbols=['XAUUSD'],
        primary_symbol='XAUUSD',
        timeframe=Timeframe.M5,
    )


@pytest.fixture()
def mock_mt5() -> MagicMock:
    """Pre-wired MagicMock standing in for the MetaTrader5 module."""
    mt5 = MagicMock()
    mt5.initialize.return_value = True
    mt5.symbol_select.return_value = True
    account = MagicMock()
    account.login = 12345
    account.server = 'TestBroker'
    account.balance = 10_000.0
    account.currency = 'USD'
    mt5.account_info.return_value = account
    return mt5


@pytest.fixture()
def provider(settings: Settings, mock_mt5: MagicMock) -> MT5Provider:
    """Connected MT5Provider with the real MT5 module replaced by a mock."""
    p = MT5Provider(settings)
    p._connected = True
    p._mt5 = mock_mt5
    return p


# ---------------------------------------------------------------------------
# get_ohlcv — range mode (start / end)
# ---------------------------------------------------------------------------


class TestGetOhlcvRangeMode:
    def test_selects_symbol_before_fetch(self, provider: MT5Provider, mock_mt5: MagicMock) -> None:
        mock_mt5.copy_rates_range.return_value = _make_rates()
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 2, tzinfo=UTC)

        provider.get_ohlcv('XAUUSD', Timeframe.M5, start=start, end=end)

        mock_mt5.symbol_select.assert_called_once_with('XAUUSDm', True)

    def test_passes_naive_datetimes_to_mt5(
        self, provider: MT5Provider, mock_mt5: MagicMock
    ) -> None:
        mock_mt5.copy_rates_range.return_value = _make_rates()
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2026, 5, 29, tzinfo=UTC)

        provider.get_ohlcv('XAUUSD', Timeframe.M5, start=start, end=end)

        # Chunking makes multiple calls; check first and last boundary dates
        all_calls = mock_mt5.copy_rates_range.call_args_list
        first_from = all_calls[0][0][2]
        last_to = all_calls[-1][0][3]
        assert first_from.tzinfo is None, 'date_from must be naive'
        assert last_to.tzinfo is None, 'date_to must be naive'
        assert first_from == datetime(2025, 1, 1)
        assert last_to == datetime(2026, 5, 29)

    def test_returns_expected_columns_and_length(
        self, provider: MT5Provider, mock_mt5: MagicMock
    ) -> None:
        # Use a range short enough to fit in a single 90-day chunk
        n = 5
        mock_mt5.copy_rates_range.return_value = _make_rates(n)
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 2, 1, tzinfo=UTC)

        df = provider.get_ohlcv('XAUUSD', Timeframe.M5, start=start, end=end)

        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert len(df) == n
        assert str(df.index.tz) == 'UTC'

    def test_defaults_end_to_now_when_omitted(
        self, provider: MT5Provider, mock_mt5: MagicMock
    ) -> None:
        mock_mt5.copy_rates_range.return_value = _make_rates()
        start = datetime(2025, 1, 1, tzinfo=UTC)

        provider.get_ohlcv('XAUUSD', Timeframe.M5, start=start)

        _, _, _, date_to = mock_mt5.copy_rates_range.call_args[0]
        assert date_to.tzinfo is None

    def test_raises_on_none_rates(self, provider: MT5Provider, mock_mt5: MagicMock) -> None:
        mock_mt5.copy_rates_range.return_value = None
        mock_mt5.last_error.return_value = (-2, 'Invalid params')
        start = datetime(2025, 1, 1, tzinfo=UTC)

        with pytest.raises(RuntimeError, match='MT5 returned no data for XAUUSDm'):
            provider.get_ohlcv('XAUUSD', Timeframe.M5, start=start)

    def test_raises_on_empty_rates(self, provider: MT5Provider, mock_mt5: MagicMock) -> None:
        mock_mt5.copy_rates_range.return_value = _make_rates(0)
        mock_mt5.last_error.return_value = (-4, 'Not found')
        start = datetime(2025, 1, 1, tzinfo=UTC)

        with pytest.raises(RuntimeError, match='MT5 returned no data'):
            provider.get_ohlcv('XAUUSD', Timeframe.M5, start=start)


# ---------------------------------------------------------------------------
# get_ohlcv — bars mode (no start)
# ---------------------------------------------------------------------------


class TestGetOhlcvBarsMode:
    def test_calls_copy_rates_from_pos(self, provider: MT5Provider, mock_mt5: MagicMock) -> None:
        n = 100
        mock_mt5.copy_rates_from_pos.return_value = _make_rates(n)

        df = provider.get_ohlcv('XAUUSD', Timeframe.M5, bars=n)

        mock_mt5.copy_rates_from_pos.assert_called_once_with('XAUUSDm', 5, 0, n)
        assert len(df) == n

    def test_selects_symbol_in_bars_mode(self, provider: MT5Provider, mock_mt5: MagicMock) -> None:
        mock_mt5.copy_rates_from_pos.return_value = _make_rates(50)

        provider.get_ohlcv('XAUUSD', Timeframe.M5, bars=50)

        mock_mt5.symbol_select.assert_called_once_with('XAUUSDm', True)


# ---------------------------------------------------------------------------
# Connection guard
# ---------------------------------------------------------------------------


class TestConnectionGuard:
    def test_raises_when_not_connected(self, settings: Settings) -> None:
        p = MT5Provider(settings)

        with pytest.raises(RuntimeError, match='not connected'):
            p.get_ohlcv('XAUUSD', Timeframe.M5)
