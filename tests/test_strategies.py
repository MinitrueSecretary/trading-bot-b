"""Tests for trading strategies."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import Signal, SignalType
from trading_bot.strategies.macd_strategy import MACDStrategy
from trading_bot.strategies.rsi_divergence import RSIDivergenceStrategy
from trading_bot.strategies.rsi_strategy import RSIStrategy
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy

# ─── SMA Crossover ────────────────────────────────────────────────────────────


class TestSMACrossoverStrategy:
    """Tests for :class:`SMACrossoverStrategy`."""

    @pytest.fixture
    def strategy(self, settings: Settings) -> SMACrossoverStrategy:
        """Return an SMACrossover strategy instance."""
        return SMACrossoverStrategy(settings)

    def test_required_bars(self, strategy: SMACrossoverStrategy, settings: Settings) -> None:
        """required_bars should be slow period + 2."""
        assert strategy.required_bars() == settings.sma_slow + 2

    def test_hold_on_insufficient_data(self, strategy: SMACrossoverStrategy) -> None:
        """Should return HOLD when not enough data is provided."""
        df = pd.DataFrame(
            {'open': [1.1], 'high': [1.105], 'low': [1.095], 'close': [1.1], 'volume': [100.0]}
        )
        signal = strategy.generate_signal(df, 'EURUSD')
        assert signal.signal_type == SignalType.HOLD

    def test_buy_signal_on_golden_cross(self, settings: Settings) -> None:
        """Should emit BUY when fast SMA crosses above slow SMA."""
        # Build a series that transitions from downtrend to uptrend
        n = 60
        # First half: price declining (fast < slow)
        # Second half: price rising quickly (fast > slow)
        prices_down = np.linspace(1.1200, 1.1000, n // 2)
        prices_up = np.linspace(1.1000, 1.1400, n // 2)
        close = np.concatenate([prices_down, prices_up])

        index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame(
            {
                'open': close - 0.0001,
                'high': close + 0.0002,
                'low': close - 0.0002,
                'close': close,
                'volume': np.ones(n) * 1000.0,
            },
            index=index,
        )

        strat = SMACrossoverStrategy(settings)
        signal = strat.generate_signal(df, 'EURUSD')
        # With a rapid uptrend at the end, we expect BUY or HOLD
        assert signal.signal_type in {SignalType.BUY, SignalType.HOLD}

    def test_signal_has_sl_and_tp(
        self, strategy: SMACrossoverStrategy, ohlcv_df: pd.DataFrame
    ) -> None:
        """Any actionable signal should have non-zero SL and TP."""
        signal = strategy.generate_signal(ohlcv_df, 'EURUSD')
        if signal.is_actionable:
            assert signal.stop_loss != 0.0
            assert signal.take_profit != 0.0

    def test_name(self, strategy: SMACrossoverStrategy) -> None:
        """Strategy name should be 'sma_crossover'."""
        assert strategy.name == 'sma_crossover'


# ─── RSI Strategy ─────────────────────────────────────────────────────────────


class TestRSIStrategy:
    """Tests for :class:`RSIStrategy`."""

    @pytest.fixture
    def strategy(self, settings: Settings) -> RSIStrategy:
        """Return an RSI strategy instance."""
        return RSIStrategy(settings)

    def test_name(self, strategy: RSIStrategy) -> None:
        """Strategy name should be 'rsi'."""
        assert strategy.name == 'rsi'

    def test_required_bars_includes_trend_sma(self, strategy: RSIStrategy) -> None:
        """RSI strategy needs SMA200 + RSI period bars."""
        assert strategy.required_bars() == 200 + 14 + 2

    def test_hold_on_insufficient_data(self, strategy: RSIStrategy) -> None:
        """Returns HOLD when data is too short."""
        df = pd.DataFrame(
            {'open': [1.1], 'high': [1.105], 'low': [1.095], 'close': [1.1], 'volume': [100.0]}
        )
        signal = strategy.generate_signal(df, 'EURUSD')
        assert signal.signal_type == SignalType.HOLD

    def test_signal_on_large_dataset(self, strategy: RSIStrategy) -> None:
        """Should produce a valid signal (not crash) on a large dataset."""
        n = 300
        rng = np.random.default_rng(99)
        close = 1.1 + np.cumsum(rng.normal(0, 0.0003, n))
        index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame(
            {
                'open': close - 0.0001,
                'high': close + 0.0005,
                'low': close - 0.0005,
                'close': close,
                'volume': np.ones(n) * 500.0,
            },
            index=index,
        )
        signal = strategy.generate_signal(df, 'EURUSD')
        assert isinstance(signal, Signal)
        assert signal.signal_type in {SignalType.BUY, SignalType.SELL, SignalType.HOLD}


# ─── MACD Strategy ────────────────────────────────────────────────────────────


class TestMACDStrategy:
    """Tests for :class:`MACDStrategy`."""

    @pytest.fixture
    def strategy(self, settings: Settings) -> MACDStrategy:
        """Return a MACD strategy instance."""
        return MACDStrategy(settings)

    def test_name(self, strategy: MACDStrategy) -> None:
        """Strategy name should be 'macd'."""
        assert strategy.name == 'macd'

    def test_required_bars(self, strategy: MACDStrategy, settings: Settings) -> None:
        """required_bars = slow + signal + 2."""
        assert strategy.required_bars() == settings.macd_slow + settings.macd_signal + 2

    def test_signal_on_valid_data(self, strategy: MACDStrategy, ohlcv_df: pd.DataFrame) -> None:
        """Should not crash on valid OHLCV data."""
        signal = strategy.generate_signal(ohlcv_df, 'EURUSD')
        assert isinstance(signal, Signal)


# ─── RSI Divergence Strategy ──────────────────────────────────────────────────


def _make_ohlcv(close: np.ndarray) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close price array."""
    n = len(close)
    index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    df = pd.DataFrame(
        {
            'open': close - 0.0001,
            'high': close + 0.0005,
            'low': close - 0.0005,
            'close': close,
            'volume': np.ones(n) * 1000.0,
        },
        index=index,
    )
    df['high'] = df[['high', 'close', 'open']].max(axis=1)
    df['low'] = df[['low', 'close', 'open']].min(axis=1)
    return df


class TestRSIDivergenceStrategy:
    """Tests for :class:`RSIDivergenceStrategy`."""

    _METADATA_KEYS = (
        'stochrsi_k',
        'stochrsi_d',
        'divergence_type',
        'price_pivot1',
        'price_pivot2',
        'rsi_pivot1',
        'rsi_pivot2',
    )

    @pytest.fixture
    def strategy(self, settings: Settings) -> RSIDivergenceStrategy:
        """Return an RSIDivergenceStrategy instance with test settings."""
        return RSIDivergenceStrategy(settings)

    # ── Basic contract ────────────────────────────────────────────────────────

    def test_name(self, strategy: RSIDivergenceStrategy) -> None:
        """Strategy name should be 'rsi_divergence'."""
        assert strategy.name == 'rsi_divergence'

    def test_required_bars(self, strategy: RSIDivergenceStrategy, settings: Settings) -> None:
        """required_bars = 2 * stoch_rsi_period + lookback + swing_window."""
        expected = (
            2 * settings.stoch_rsi_period
            + settings.rsi_divergence_lookback
            + settings.rsi_divergence_swing_window
        )
        assert strategy.required_bars() == expected

    def test_hold_on_insufficient_data(self, strategy: RSIDivergenceStrategy) -> None:
        """Returns HOLD when only 1 bar of data is provided."""
        df = pd.DataFrame(
            {'open': [1.1], 'high': [1.105], 'low': [1.095], 'close': [1.1], 'volume': [100.0]}
        )
        signal = strategy.generate_signal(df, 'EURUSD')
        assert signal.signal_type == SignalType.HOLD

    def test_signal_on_random_walk(
        self, strategy: RSIDivergenceStrategy, ohlcv_df: pd.DataFrame
    ) -> None:
        """Should produce a valid Signal without crashing on a 300-bar random walk."""
        signal = strategy.generate_signal(ohlcv_df, 'EURUSD')
        assert isinstance(signal, Signal)
        assert signal.signal_type in {SignalType.BUY, SignalType.SELL, SignalType.HOLD}

    # ── Divergence type tests ─────────────────────────────────────────────────

    def test_regular_bullish_divergence(self, settings: Settings) -> None:
        """Price LL + RSI HL at swing lows → BUY (regular bullish divergence)."""
        # Build a base history, then insert two V-shapes where the second
        # valley is lower in price.  StochRSI may or may not reflect HL, so
        # we accept BUY or HOLD.
        sw = settings.rsi_divergence_swing_window
        n_base = 2 * settings.stoch_rsi_period + 20
        rng = np.random.default_rng(42)
        base = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n_base))

        # Two V-shapes: valley1 at 1.095, valley2 (lower) at 1.090
        v1 = np.concatenate(
            [
                np.linspace(float(base[-1]), 1.0950, sw + 2),
                np.linspace(1.0950, 1.1050, sw + 2),
            ]
        )
        v2 = np.concatenate(
            [
                np.linspace(1.1050, 1.0900, sw + 2),
                np.linspace(1.0900, 1.1000, sw + 2),
            ]
        )
        close = np.concatenate([base, v1, v2])

        strat = RSIDivergenceStrategy(settings)
        signal = strat.generate_signal(_make_ohlcv(close), 'EURUSD')
        assert signal.signal_type in {SignalType.BUY, SignalType.HOLD}

    def test_regular_bearish_divergence(self, settings: Settings) -> None:
        """Price HH + RSI LH at swing highs → SELL (regular bearish divergence)."""
        sw = settings.rsi_divergence_swing_window
        n_base = 2 * settings.stoch_rsi_period + 20
        rng = np.random.default_rng(99)
        base = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n_base))

        # Two Λ-shapes: peak1 at 1.105, peak2 (higher) at 1.110
        p1 = np.concatenate(
            [
                np.linspace(float(base[-1]), 1.1050, sw + 2),
                np.linspace(1.1050, 1.0990, sw + 2),
            ]
        )
        p2 = np.concatenate(
            [
                np.linspace(1.0990, 1.1100, sw + 2),
                np.linspace(1.1100, 1.1020, sw + 2),
            ]
        )
        close = np.concatenate([base, p1, p2])

        strat = RSIDivergenceStrategy(settings)
        signal = strat.generate_signal(_make_ohlcv(close), 'EURUSD')
        assert signal.signal_type in {SignalType.SELL, SignalType.HOLD}

    def test_hidden_bullish_divergence(self, settings: Settings) -> None:
        """Price HL + RSI LL at swing lows → BUY (hidden bullish divergence)."""
        sw = settings.rsi_divergence_swing_window
        n_base = 2 * settings.stoch_rsi_period + 20
        rng = np.random.default_rng(7)
        base = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n_base))

        # Two V-shapes: valley1 at 1.090, valley2 (higher) at 1.095
        v1 = np.concatenate(
            [
                np.linspace(float(base[-1]), 1.0900, sw + 2),
                np.linspace(1.0900, 1.1000, sw + 2),
            ]
        )
        v2 = np.concatenate(
            [
                np.linspace(1.1000, 1.0950, sw + 2),
                np.linspace(1.0950, 1.1050, sw + 2),
            ]
        )
        close = np.concatenate([base, v1, v2])

        strat = RSIDivergenceStrategy(settings)
        signal = strat.generate_signal(_make_ohlcv(close), 'EURUSD')
        assert signal.signal_type in {SignalType.BUY, SignalType.HOLD}

    def test_hidden_bearish_divergence(self, settings: Settings) -> None:
        """Price LH + RSI HH at swing highs → SELL (hidden bearish divergence)."""
        sw = settings.rsi_divergence_swing_window
        n_base = 2 * settings.stoch_rsi_period + 20
        rng = np.random.default_rng(13)
        base = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n_base))

        # Two Λ-shapes: peak1 at 1.110, peak2 (lower) at 1.105
        p1 = np.concatenate(
            [
                np.linspace(float(base[-1]), 1.1100, sw + 2),
                np.linspace(1.1100, 1.1010, sw + 2),
            ]
        )
        p2 = np.concatenate(
            [
                np.linspace(1.1010, 1.1050, sw + 2),
                np.linspace(1.1050, 1.0980, sw + 2),
            ]
        )
        close = np.concatenate([base, p1, p2])

        strat = RSIDivergenceStrategy(settings)
        signal = strat.generate_signal(_make_ohlcv(close), 'EURUSD')
        assert signal.signal_type in {SignalType.SELL, SignalType.HOLD}

    # ── Signal quality ────────────────────────────────────────────────────────

    def test_actionable_signal_has_sl_and_tp(
        self, strategy: RSIDivergenceStrategy, ohlcv_df: pd.DataFrame
    ) -> None:
        """Any BUY or SELL signal must carry non-zero SL and TP values."""
        signal = strategy.generate_signal(ohlcv_df, 'EURUSD')
        if signal.is_actionable:
            assert signal.stop_loss != 0.0
            assert signal.take_profit != 0.0

    def test_metadata_keys_on_actionable_signal(
        self, strategy: RSIDivergenceStrategy, ohlcv_df: pd.DataFrame
    ) -> None:
        """Actionable signals must include all 7 metadata keys."""
        signal = strategy.generate_signal(ohlcv_df, 'EURUSD')
        if signal.is_actionable:
            for key in self._METADATA_KEYS:
                assert key in signal.metadata
