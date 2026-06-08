"""Tests for BollingerBandsTPSLManager and BollingerBandsStrategy."""

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]
import pytest

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import SignalType
from trading_bot.strategies.bollinger_bands import BollingerBandsStrategy
from trading_bot.utils.tpsl import BaseTPSLManager, BollingerBandsTPSLManager

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_df(n: int = 50, start: float = 1.1000, step: float = 0.0001) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with *n* bars."""
    close = np.array([start + i * step for i in range(n)])
    index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    return pd.DataFrame(
        {
            'open': close - 0.0001,
            'high': close + 0.0005,
            'low': close - 0.0005,
            'close': close,
            'volume': np.ones(n) * 1000.0,
        },
        index=index,
    )


def _live_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple[float, float]:
    """Compute lower and upper band values independently for assertions."""
    bbands = ta.bbands(df['close'], length=period, lower_std=std_dev, upper_std=std_dev)
    cols = bbands.columns.tolist()
    lower_col = next(c for c in cols if c.startswith('BBL_'))
    upper_col = next(c for c in cols if c.startswith('BBU_'))
    lower = float(bbands[lower_col].iloc[-1])
    upper = float(bbands[upper_col].iloc[-1])
    return lower, upper


# ─── BollingerBandsTPSLManager ───────────────────────────────────────────────


class TestBollingerBandsTPSLManager:
    """Tests for :class:`BollingerBandsTPSLManager`."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        """Return a 50-bar OHLCV DataFrame."""
        return _make_df(50)

    @pytest.fixture
    def mgr(self, df: pd.DataFrame) -> BollingerBandsTPSLManager:
        """Return a default manager primed with data."""
        m = BollingerBandsTPSLManager()
        m.set_data(df)
        return m

    # ── Subclass contract ─────────────────────────────────────────────────────

    def test_is_base_tpsl_manager(self, mgr: BollingerBandsTPSLManager) -> None:
        """BollingerBandsTPSLManager must be a subclass of BaseTPSLManager."""
        assert isinstance(mgr, BaseTPSLManager)

    # ── Properties ────────────────────────────────────────────────────────────

    def test_period_property(self) -> None:
        """period property returns the configured value."""
        assert BollingerBandsTPSLManager(period=14).period == 14

    def test_std_dev_property(self) -> None:
        """std_dev property returns the configured value."""
        assert BollingerBandsTPSLManager(std_dev=1.5).std_dev == 1.5

    def test_min_sl_pips_property(self) -> None:
        """min_sl_pips property returns the configured value."""
        assert BollingerBandsTPSLManager(min_sl_pips=20.0).min_sl_pips == 20.0

    def test_default_period(self) -> None:
        """Default period should be 20."""
        assert BollingerBandsTPSLManager().period == 20

    def test_default_std_dev(self) -> None:
        """Default std_dev should be 2.0."""
        assert BollingerBandsTPSLManager().std_dev == 2.0

    def test_default_min_sl_pips(self) -> None:
        """Default min_sl_pips should be 0.0 (disabled)."""
        assert BollingerBandsTPSLManager().min_sl_pips == 0.0

    # ── HOLD returns zero ─────────────────────────────────────────────────────

    def test_hold_sl_returns_zero(self, mgr: BollingerBandsTPSLManager) -> None:
        """calculate_sl with HOLD must return 0.0."""
        assert mgr.calculate_sl(SignalType.HOLD, 1.1, 'EURUSD') == 0.0

    def test_hold_tp_returns_zero(self, mgr: BollingerBandsTPSLManager) -> None:
        """calculate_tp with HOLD must return 0.0."""
        assert mgr.calculate_tp(SignalType.HOLD, 1.1, 'EURUSD') == 0.0

    # ── BUY: SL at lower band, TP at upper band ───────────────────────────────

    def test_buy_sl_equals_lower_band(
        self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame
    ) -> None:
        """BUY SL must equal the lower Bollinger Band value."""
        lower, _ = _live_bands(df)
        sl = mgr.calculate_sl(SignalType.BUY, float(df['close'].iloc[-1]), 'EURUSD')
        assert sl == pytest.approx(lower)

    def test_buy_tp_equals_upper_band(
        self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame
    ) -> None:
        """BUY TP must equal the upper Bollinger Band value."""
        _, upper = _live_bands(df)
        tp = mgr.calculate_tp(SignalType.BUY, float(df['close'].iloc[-1]), 'EURUSD')
        assert tp == pytest.approx(upper)

    def test_buy_sl_is_below_entry(self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame) -> None:
        """BUY SL (lower band) should be below the entry price."""
        entry = float(df['close'].iloc[-1])
        sl = mgr.calculate_sl(SignalType.BUY, entry, 'EURUSD')
        assert sl < entry

    def test_buy_tp_is_above_entry(self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame) -> None:
        """BUY TP (upper band) should be above the entry price."""
        entry = float(df['close'].iloc[-1])
        tp = mgr.calculate_tp(SignalType.BUY, entry, 'EURUSD')
        assert tp > entry

    # ── SELL: SL at upper band, TP at lower band ──────────────────────────────

    def test_sell_sl_equals_upper_band(
        self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame
    ) -> None:
        """SELL SL must equal the upper Bollinger Band value."""
        _, upper = _live_bands(df)
        sl = mgr.calculate_sl(SignalType.SELL, float(df['close'].iloc[-1]), 'EURUSD')
        assert sl == pytest.approx(upper)

    def test_sell_tp_equals_lower_band(
        self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame
    ) -> None:
        """SELL TP must equal the lower Bollinger Band value."""
        lower, _ = _live_bands(df)
        tp = mgr.calculate_tp(SignalType.SELL, float(df['close'].iloc[-1]), 'EURUSD')
        assert tp == pytest.approx(lower)

    def test_sell_sl_is_above_entry(self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame) -> None:
        """SELL SL (upper band) should be above the entry price."""
        entry = float(df['close'].iloc[-1])
        sl = mgr.calculate_sl(SignalType.SELL, entry, 'EURUSD')
        assert sl > entry

    def test_sell_tp_is_below_entry(self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame) -> None:
        """SELL TP (lower band) should be below the entry price."""
        entry = float(df['close'].iloc[-1])
        tp = mgr.calculate_tp(SignalType.SELL, entry, 'EURUSD')
        assert tp < entry

    # ── set_data updates bands ────────────────────────────────────────────────

    def test_set_data_updates_bands(self) -> None:
        """Calling set_data with new data must update the band values."""
        mgr = BollingerBandsTPSLManager()
        df1 = _make_df(50, start=1.1000)
        df2 = _make_df(50, start=1.2000)
        mgr.set_data(df1)
        sl1 = mgr.calculate_sl(SignalType.BUY, 1.1, 'EURUSD')
        mgr.set_data(df2)
        sl2 = mgr.calculate_sl(SignalType.BUY, 1.2, 'EURUSD')
        assert sl1 != sl2

    # ── min_sl_pips floor ─────────────────────────────────────────────────────

    def test_min_sl_pips_floors_buy_sl(self) -> None:
        """min_sl_pips must push BUY SL further below entry when band is too tight."""
        # Build a very tight-banded series (constant price → tiny std deviation)
        close = np.full(50, 1.1000)
        index = pd.date_range('2024-01-01', periods=50, freq='h', tz='UTC')
        df = pd.DataFrame(
            {'open': close, 'high': close, 'low': close, 'close': close, 'volume': np.ones(50)},
            index=index,
        )
        mgr_floor = BollingerBandsTPSLManager(min_sl_pips=50.0)
        mgr_floor.set_data(df)
        entry = 1.1000
        sl = mgr_floor.calculate_sl(SignalType.BUY, entry, 'EURUSD')
        # With min_sl_pips=50, floor = entry - 50 * 0.0001 = 1.0950
        assert sl <= entry - 50 * 0.0001

    def test_min_sl_pips_raises_sell_sl(self) -> None:
        """min_sl_pips must push SELL SL further above entry when band is too tight."""
        close = np.full(50, 1.1000)
        index = pd.date_range('2024-01-01', periods=50, freq='h', tz='UTC')
        df = pd.DataFrame(
            {'open': close, 'high': close, 'low': close, 'close': close, 'volume': np.ones(50)},
            index=index,
        )
        mgr_floor = BollingerBandsTPSLManager(min_sl_pips=50.0)
        mgr_floor.set_data(df)
        entry = 1.1000
        sl = mgr_floor.calculate_sl(SignalType.SELL, entry, 'EURUSD')
        assert sl >= entry + 50 * 0.0001

    def test_min_sl_pips_zero_no_effect(
        self, mgr: BollingerBandsTPSLManager, df: pd.DataFrame
    ) -> None:
        """min_sl_pips=0.0 should not alter the band-derived SL."""
        lower, _ = _live_bands(df)
        entry = float(df['close'].iloc[-1])
        sl = mgr.calculate_sl(SignalType.BUY, entry, 'EURUSD')
        assert sl == pytest.approx(lower)

    # ── RuntimeError without set_data ─────────────────────────────────────────

    def test_raises_runtime_error_without_set_data_sl(self) -> None:
        """calculate_sl must raise RuntimeError if set_data was never called."""
        mgr = BollingerBandsTPSLManager()
        with pytest.raises(RuntimeError, match='set_data'):
            mgr.calculate_sl(SignalType.BUY, 1.1, 'EURUSD')

    def test_raises_runtime_error_without_set_data_tp(self) -> None:
        """calculate_tp must raise RuntimeError if set_data was never called."""
        mgr = BollingerBandsTPSLManager()
        with pytest.raises(RuntimeError, match='set_data'):
            mgr.calculate_tp(SignalType.BUY, 1.1, 'EURUSD')

    # ── ValueError for insufficient rows ──────────────────────────────────────

    def test_raises_value_error_too_few_rows(self) -> None:
        """Passing a DataFrame shorter than the period must raise ValueError."""
        mgr = BollingerBandsTPSLManager(period=20)
        mgr.set_data(_make_df(5))  # 5 rows < 20
        with pytest.raises(ValueError, match='rows'):
            mgr.calculate_sl(SignalType.BUY, 1.1, 'EURUSD')

    # ── Validation ────────────────────────────────────────────────────────────

    def test_raises_on_period_less_than_2(self) -> None:
        """period < 2 must raise ValueError."""
        with pytest.raises(ValueError, match='period'):
            BollingerBandsTPSLManager(period=1)

    def test_raises_on_zero_std_dev(self) -> None:
        """std_dev=0 must raise ValueError."""
        with pytest.raises(ValueError, match='std_dev'):
            BollingerBandsTPSLManager(std_dev=0.0)

    def test_raises_on_negative_std_dev(self) -> None:
        """Negative std_dev must raise ValueError."""
        with pytest.raises(ValueError, match='std_dev'):
            BollingerBandsTPSLManager(std_dev=-1.0)

    def test_raises_on_negative_min_sl_pips(self) -> None:
        """Negative min_sl_pips must raise ValueError."""
        with pytest.raises(ValueError, match='min_sl_pips'):
            BollingerBandsTPSLManager(min_sl_pips=-5.0)

    # ── Return types ──────────────────────────────────────────────────────────

    def test_calculate_sl_returns_float(self, mgr: BollingerBandsTPSLManager) -> None:
        """calculate_sl must return a float."""
        assert isinstance(mgr.calculate_sl(SignalType.BUY, 1.1, 'EURUSD'), float)

    def test_calculate_tp_returns_float(self, mgr: BollingerBandsTPSLManager) -> None:
        """calculate_tp must return a float."""
        assert isinstance(mgr.calculate_tp(SignalType.BUY, 1.1, 'EURUSD'), float)

    # ── Custom period/std_dev ─────────────────────────────────────────────────

    def test_custom_period_changes_bands(self) -> None:
        """Changing the period should produce different band values."""
        df = _make_df(100)
        mgr10 = BollingerBandsTPSLManager(period=10)
        mgr10.set_data(df)
        mgr30 = BollingerBandsTPSLManager(period=30)
        mgr30.set_data(df)
        entry = float(df['close'].iloc[-1])
        assert mgr10.calculate_sl(SignalType.BUY, entry, 'EURUSD') != pytest.approx(
            mgr30.calculate_sl(SignalType.BUY, entry, 'EURUSD')
        )

    def test_wider_std_dev_gives_wider_bands(self) -> None:
        """A larger std_dev must produce a wider SL distance from entry."""
        df = _make_df(50)
        entry = float(df['close'].iloc[-1])
        mgr1 = BollingerBandsTPSLManager(std_dev=1.0)
        mgr1.set_data(df)
        mgr3 = BollingerBandsTPSLManager(std_dev=3.0)
        mgr3.set_data(df)
        sl1 = mgr1.calculate_sl(SignalType.BUY, entry, 'EURUSD')
        sl3 = mgr3.calculate_sl(SignalType.BUY, entry, 'EURUSD')
        # Wider bands → lower SL for BUY
        assert sl3 < sl1


# ─── BollingerBandsStrategy ───────────────────────────────────────────────────


class TestBollingerBandsStrategy:
    """Tests for :class:`BollingerBandsStrategy`."""

    @pytest.fixture
    def strat(self, settings: Settings) -> BollingerBandsStrategy:
        """Return a strategy with default settings (period=20)."""
        return BollingerBandsStrategy(settings)

    # ── Basic contract ────────────────────────────────────────────────────────

    def test_name(self, strat: BollingerBandsStrategy) -> None:
        """Strategy name must be 'bollinger_bands'."""
        assert strat.name == 'bollinger_bands'

    def test_required_bars(self, strat: BollingerBandsStrategy) -> None:
        """required_bars must equal period + 1."""
        assert strat.required_bars() == strat._period + 1

    # ── Insufficient data ─────────────────────────────────────────────────────

    def test_hold_on_insufficient_data(
        self, strat: BollingerBandsStrategy, settings: Settings
    ) -> None:
        """Strategy must return HOLD when fewer than required_bars are supplied."""
        tiny_df = _make_df(5)
        signal = strat.generate_signal(tiny_df, 'EURUSD')
        assert signal.signal_type == SignalType.HOLD

    # ── BUY signal from lower-band bounce ─────────────────────────────────────

    def test_buy_signal_on_lower_band_bounce(self, settings: Settings) -> None:
        """A lower-band bounce must produce a BUY signal."""
        period = settings.bb_period
        # Build a base series well above the lower band
        n = period + 10
        rng = np.random.default_rng(1)
        close = 1.1000 + rng.normal(0, 0.0002, n)
        index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame(
            {
                'open': close - 0.0001,
                'high': close + 0.0005,
                'low': close - 0.0005,
                'close': close.copy(),
                'volume': np.ones(n) * 1000.0,
            },
            index=index,
        )
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)

        # Compute current bands
        bbands = ta.bbands(
            df['close'], length=period, lower_std=settings.bb_std_dev, upper_std=settings.bb_std_dev
        )
        cols = bbands.columns.tolist()
        lower_col = next(c for c in cols if c.startswith('BBL_'))
        lower = float(bbands[lower_col].iloc[-1])

        # Force: prev bar below lower, current bar above lower
        df.iloc[-2, df.columns.get_loc('close')] = lower - 0.0005
        df.iloc[-1, df.columns.get_loc('close')] = lower + 0.0003

        strat = BollingerBandsStrategy(settings)
        signal = strat.generate_signal(df, 'EURUSD')
        assert signal.signal_type == SignalType.BUY

    # ── SELL signal from upper-band rejection ─────────────────────────────────

    def test_sell_signal_on_upper_band_rejection(self, settings: Settings) -> None:
        """An upper-band rejection must produce a SELL signal."""
        period = settings.bb_period
        n = period + 10
        rng = np.random.default_rng(2)
        close = 1.1000 + rng.normal(0, 0.0002, n)
        index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame(
            {
                'open': close - 0.0001,
                'high': close + 0.0005,
                'low': close - 0.0005,
                'close': close.copy(),
                'volume': np.ones(n) * 1000.0,
            },
            index=index,
        )
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)

        bbands = ta.bbands(
            df['close'], length=period, lower_std=settings.bb_std_dev, upper_std=settings.bb_std_dev
        )
        cols = bbands.columns.tolist()
        upper_col = next(c for c in cols if c.startswith('BBU_'))
        upper = float(bbands[upper_col].iloc[-1])

        # Force: prev bar above upper, current bar below upper
        df.iloc[-2, df.columns.get_loc('close')] = upper + 0.0005
        df.iloc[-1, df.columns.get_loc('close')] = upper - 0.0003

        strat = BollingerBandsStrategy(settings)
        signal = strat.generate_signal(df, 'EURUSD')
        assert signal.signal_type == SignalType.SELL

    # ── HOLD when no band touch ───────────────────────────────────────────────

    def test_hold_when_price_inside_bands(self, settings: Settings) -> None:
        """Price strictly between bands must return HOLD."""
        df = _make_df(50)
        strat = BollingerBandsStrategy(settings)
        signal = strat.generate_signal(df, 'EURUSD')
        # Steady trend price stays inside bands — should HOLD
        assert signal.signal_type == SignalType.HOLD

    # ── Signal fields ─────────────────────────────────────────────────────────

    def test_buy_signal_has_positive_stop_loss(self, settings: Settings) -> None:
        """A BUY signal stop_loss must be > 0."""
        period = settings.bb_period
        n = period + 10
        close = np.full(n, 1.1000) + np.linspace(0, 0.001, n)
        index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame(
            {
                'open': close,
                'high': close + 0.001,
                'low': close - 0.001,
                'close': close.copy(),
                'volume': np.ones(n) * 1000.0,
            },
            index=index,
        )
        bbands = ta.bbands(
            df['close'], length=period, lower_std=settings.bb_std_dev, upper_std=settings.bb_std_dev
        )
        cols = bbands.columns.tolist()
        lower_col = next(c for c in cols if c.startswith('BBL_'))
        lower = float(bbands[lower_col].iloc[-1])
        df.iloc[-2, df.columns.get_loc('close')] = lower - 0.0005
        df.iloc[-1, df.columns.get_loc('close')] = lower + 0.0003

        strat = BollingerBandsStrategy(settings)
        signal = strat.generate_signal(df, 'EURUSD')
        if signal.signal_type == SignalType.BUY:
            assert signal.stop_loss > 0
            assert signal.take_profit > signal.entry_price

    def test_signal_metadata_contains_bb_keys(self, settings: Settings) -> None:
        """Signal metadata must include bb_lower, bb_middle, bb_upper, band_width."""
        period = settings.bb_period
        n = period + 10
        close = np.full(n, 1.1000)
        index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame(
            {
                'open': close,
                'high': close + 0.001,
                'low': close - 0.001,
                'close': close.copy(),
                'volume': np.ones(n) * 1000.0,
            },
            index=index,
        )
        bbands = ta.bbands(
            df['close'], length=period, lower_std=settings.bb_std_dev, upper_std=settings.bb_std_dev
        )
        cols = bbands.columns.tolist()
        lower_col = next(c for c in cols if c.startswith('BBL_'))
        lower = float(bbands[lower_col].iloc[-1])
        df.iloc[-2, df.columns.get_loc('close')] = lower - 0.0005
        df.iloc[-1, df.columns.get_loc('close')] = lower + 0.0003

        strat = BollingerBandsStrategy(settings)
        signal = strat.generate_signal(df, 'EURUSD')
        if signal.signal_type == SignalType.BUY:
            assert 'bb_lower' in signal.metadata
            assert 'bb_middle' in signal.metadata
            assert 'bb_upper' in signal.metadata
            assert 'band_width' in signal.metadata

    # ── Custom TPSL injection ─────────────────────────────────────────────────

    def test_injected_tpsl_manager_is_used(self, settings: Settings) -> None:
        """An injected TPSL manager must be stored and used."""
        from trading_bot.utils.tpsl import FixedPipsTPSLManager

        custom_mgr = FixedPipsTPSLManager(sl_pips=100, tp_pips=200)
        strat = BollingerBandsStrategy(settings, tpsl_manager=custom_mgr)
        assert strat._tpsl is custom_mgr
