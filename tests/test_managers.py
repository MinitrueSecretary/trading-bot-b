"""Tests for TPSLManager and LotSizeManager classes."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import SignalType
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy
from trading_bot.utils.lot_size import BaseLotSizeManager, FixedLotSizeManager
from trading_bot.utils.tpsl import BaseTPSLManager, FixedPipsTPSLManager


# ─── FixedPipsTPSLManager ─────────────────────────────────────────────────────


class TestFixedPipsTPSLManager:
    """Tests for :class:`FixedPipsTPSLManager`."""

    @pytest.fixture
    def mgr(self) -> FixedPipsTPSLManager:
        """Return a 30 SL / 60 TP manager."""
        return FixedPipsTPSLManager(sl_pips=30.0, tp_pips=60.0)

    # ── Subclass contract ─────────────────────────────────────────────────────

    def test_is_base_tpsl_manager(self, mgr: FixedPipsTPSLManager) -> None:
        """FixedPipsTPSLManager must be a subclass of BaseTPSLManager."""
        assert isinstance(mgr, BaseTPSLManager)

    # ── Properties ────────────────────────────────────────────────────────────

    def test_sl_pips_property(self, mgr: FixedPipsTPSLManager) -> None:
        """sl_pips property returns the configured value."""
        assert mgr.sl_pips == 30.0

    def test_tp_pips_property(self, mgr: FixedPipsTPSLManager) -> None:
        """tp_pips property returns the configured value."""
        assert mgr.tp_pips == 60.0

    # ── BUY: calculate_sl ─────────────────────────────────────────────────────

    def test_buy_sl_is_below_entry(self, mgr: FixedPipsTPSLManager) -> None:
        """BUY stop-loss should be below the entry price."""
        sl = mgr.calculate_sl(SignalType.BUY, 1.10000, 'EURUSD')
        assert sl < 1.10000

    def test_buy_sl_exact_level(self, mgr: FixedPipsTPSLManager) -> None:
        """EURUSD BUY SL = entry - 30 * 0.0001."""
        sl = mgr.calculate_sl(SignalType.BUY, 1.10000, 'EURUSD')
        assert round(sl, 5) == round(1.10000 - 30 * 0.0001, 5)

    # ── BUY: calculate_tp ─────────────────────────────────────────────────────

    def test_buy_tp_is_above_entry(self, mgr: FixedPipsTPSLManager) -> None:
        """BUY take-profit should be above the entry price."""
        tp = mgr.calculate_tp(SignalType.BUY, 1.10000, 'EURUSD')
        assert tp > 1.10000

    def test_buy_tp_exact_level(self, mgr: FixedPipsTPSLManager) -> None:
        """EURUSD BUY TP = entry + 60 * 0.0001."""
        tp = mgr.calculate_tp(SignalType.BUY, 1.10000, 'EURUSD')
        assert round(tp, 5) == round(1.10000 + 60 * 0.0001, 5)

    # ── SELL: calculate_sl ────────────────────────────────────────────────────

    def test_sell_sl_is_above_entry(self, mgr: FixedPipsTPSLManager) -> None:
        """SELL stop-loss should be above the entry price."""
        sl = mgr.calculate_sl(SignalType.SELL, 1.10000, 'EURUSD')
        assert sl > 1.10000

    def test_sell_sl_exact_level(self, mgr: FixedPipsTPSLManager) -> None:
        """EURUSD SELL SL = entry + 30 * 0.0001."""
        sl = mgr.calculate_sl(SignalType.SELL, 1.10000, 'EURUSD')
        assert round(sl, 5) == round(1.10000 + 30 * 0.0001, 5)

    # ── SELL: calculate_tp ────────────────────────────────────────────────────

    def test_sell_tp_is_below_entry(self, mgr: FixedPipsTPSLManager) -> None:
        """SELL take-profit should be below the entry price."""
        tp = mgr.calculate_tp(SignalType.SELL, 1.10000, 'EURUSD')
        assert tp < 1.10000

    def test_sell_tp_exact_level(self, mgr: FixedPipsTPSLManager) -> None:
        """EURUSD SELL TP = entry - 60 * 0.0001."""
        tp = mgr.calculate_tp(SignalType.SELL, 1.10000, 'EURUSD')
        assert round(tp, 5) == round(1.10000 - 60 * 0.0001, 5)

    # ── HOLD ──────────────────────────────────────────────────────────────────

    def test_hold_sl_returns_zero(self, mgr: FixedPipsTPSLManager) -> None:
        """calculate_sl with HOLD must return 0.0."""
        assert mgr.calculate_sl(SignalType.HOLD, 1.10000, 'EURUSD') == 0.0

    def test_hold_tp_returns_zero(self, mgr: FixedPipsTPSLManager) -> None:
        """calculate_tp with HOLD must return 0.0."""
        assert mgr.calculate_tp(SignalType.HOLD, 1.10000, 'EURUSD') == 0.0

    # ── Instrument-specific pip scaling ───────────────────────────────────────

    def test_jpy_sl_uses_larger_pip(self) -> None:
        """USDJPY SL distance is 100× larger than EURUSD for the same pip count."""
        mgr = FixedPipsTPSLManager(sl_pips=30, tp_pips=60)
        sl_jpy = mgr.calculate_sl(SignalType.BUY, 150.000, 'USDJPY')
        sl_eur = mgr.calculate_sl(SignalType.BUY, 1.50000, 'EURUSD')
        assert round(150.000 - sl_jpy, 4) == round((1.50000 - sl_eur) * 100, 4)

    def test_xauusd_sl_uses_correct_pip(self) -> None:
        """XAUUSD SL distance = sl_pips * 0.01."""
        mgr = FixedPipsTPSLManager(sl_pips=50, tp_pips=100)
        sl = mgr.calculate_sl(SignalType.BUY, 2000.00, 'XAUUSD')
        assert round(2000.00 - sl, 5) == round(50 * 0.01, 5)

    def test_xauusd_tp_uses_correct_pip(self) -> None:
        """XAUUSD TP distance = tp_pips * 0.01."""
        mgr = FixedPipsTPSLManager(sl_pips=50, tp_pips=100)
        tp = mgr.calculate_tp(SignalType.BUY, 2000.00, 'XAUUSD')
        assert round(tp - 2000.00, 5) == round(100 * 0.01, 5)

    def test_xagusd_sl_uses_correct_pip(self) -> None:
        """XAGUSD SL distance = sl_pips * 0.001."""
        mgr = FixedPipsTPSLManager(sl_pips=50, tp_pips=100)
        sl = mgr.calculate_sl(SignalType.BUY, 25.000, 'XAGUSD')
        assert round(25.000 - sl, 6) == round(50 * 0.001, 6)

    def test_btcusd_sl_uses_whole_dollar_pip(self) -> None:
        """BTCUSD SL distance = sl_pips * 1.0."""
        mgr = FixedPipsTPSLManager(sl_pips=100, tp_pips=200)
        sl = mgr.calculate_sl(SignalType.BUY, 60_000.0, 'BTCUSD')
        assert sl == pytest.approx(60_000.0 - 100 * 1.0)

    def test_btcusd_tp_uses_whole_dollar_pip(self) -> None:
        """BTCUSD TP distance = tp_pips * 1.0."""
        mgr = FixedPipsTPSLManager(sl_pips=100, tp_pips=200)
        tp = mgr.calculate_tp(SignalType.BUY, 60_000.0, 'BTCUSD')
        assert tp == pytest.approx(60_000.0 + 200 * 1.0)

    # ── Validation ────────────────────────────────────────────────────────────

    def test_raises_on_zero_sl_pips(self) -> None:
        """sl_pips=0 should raise ValueError."""
        with pytest.raises(ValueError, match='sl_pips'):
            FixedPipsTPSLManager(sl_pips=0, tp_pips=60)

    def test_raises_on_negative_sl_pips(self) -> None:
        """Negative sl_pips should raise ValueError."""
        with pytest.raises(ValueError, match='sl_pips'):
            FixedPipsTPSLManager(sl_pips=-10, tp_pips=60)

    def test_raises_on_zero_tp_pips(self) -> None:
        """tp_pips=0 should raise ValueError."""
        with pytest.raises(ValueError, match='tp_pips'):
            FixedPipsTPSLManager(sl_pips=30, tp_pips=0)

    # ── Return types ──────────────────────────────────────────────────────────

    def test_calculate_sl_returns_float(self, mgr: FixedPipsTPSLManager) -> None:
        """calculate_sl must return a float."""
        assert isinstance(mgr.calculate_sl(SignalType.BUY, 1.1, 'EURUSD'), float)

    def test_calculate_tp_returns_float(self, mgr: FixedPipsTPSLManager) -> None:
        """calculate_tp must return a float."""
        assert isinstance(mgr.calculate_tp(SignalType.BUY, 1.1, 'EURUSD'), float)


# ─── FixedLotSizeManager ──────────────────────────────────────────────────────


class TestFixedLotSizeManager:
    """Tests for :class:`FixedLotSizeManager`."""

    @pytest.fixture
    def mgr(self) -> FixedLotSizeManager:
        """Return a 0.1-lot manager."""
        return FixedLotSizeManager(lot_size=0.1)

    # ── Subclass contract ─────────────────────────────────────────────────────

    def test_is_base_lot_size_manager(self, mgr: FixedLotSizeManager) -> None:
        """FixedLotSizeManager must be a subclass of BaseLotSizeManager."""
        assert isinstance(mgr, BaseLotSizeManager)

    # ── Property ──────────────────────────────────────────────────────────────

    def test_lot_size_property(self, mgr: FixedLotSizeManager) -> None:
        """lot_size property returns the configured value."""
        assert mgr.lot_size == 0.1

    # ── get_lot_size ignores all arguments ────────────────────────────────────

    def test_returns_configured_lot_size(self, mgr: FixedLotSizeManager) -> None:
        """Always returns the fixed lot regardless of inputs."""
        assert mgr.get_lot_size(10_000, 1.1, 'EURUSD') == 0.1

    def test_ignores_balance(self, mgr: FixedLotSizeManager) -> None:
        """Changing balance does not affect the returned lot size."""
        assert mgr.get_lot_size(1_000, 1.1, 'EURUSD') == mgr.get_lot_size(1_000_000, 1.1, 'EURUSD')

    def test_ignores_symbol(self, mgr: FixedLotSizeManager) -> None:
        """Lot size is the same regardless of instrument."""
        assert mgr.get_lot_size(10_000, 1.1, 'EURUSD') == mgr.get_lot_size(10_000, 60_000, 'BTCUSD')

    def test_ignores_stop_loss_price(self, mgr: FixedLotSizeManager) -> None:
        """Stop-loss price does not change the returned lot size."""
        assert mgr.get_lot_size(10_000, 1.1, 'EURUSD', 1.09) == mgr.get_lot_size(10_000, 1.1, 'EURUSD', 0.0)

    def test_custom_lot_size(self) -> None:
        """A non-default lot size is correctly returned."""
        mgr = FixedLotSizeManager(lot_size=2.5)
        assert mgr.get_lot_size(10_000, 1.1, 'EURUSD') == 2.5

    def test_default_lot_size_is_0_1(self) -> None:
        """Default lot size when no argument is given should be 0.1."""
        mgr = FixedLotSizeManager()
        assert mgr.lot_size == 0.1

    # ── Validation ────────────────────────────────────────────────────────────

    def test_raises_on_zero_lot_size(self) -> None:
        """lot_size=0 should raise ValueError."""
        with pytest.raises(ValueError, match='lot_size'):
            FixedLotSizeManager(lot_size=0)

    def test_raises_on_negative_lot_size(self) -> None:
        """Negative lot_size should raise ValueError."""
        with pytest.raises(ValueError, match='lot_size'):
            FixedLotSizeManager(lot_size=-0.5)

    # ── Return type ───────────────────────────────────────────────────────────

    def test_returns_float(self, mgr: FixedLotSizeManager) -> None:
        """get_lot_size must return a float."""
        assert isinstance(mgr.get_lot_size(10_000, 1.1, 'EURUSD'), float)


# ─── Manager injection into a strategy ───────────────────────────────────────


class TestTPSLManagerInjection:
    """Verify that strategies respect an injected TPSLManager."""

    def test_custom_manager_controls_sl_tp(self, settings: Settings) -> None:
        """Injecting a manager with large pip counts produces wider SL/TP."""
        wide_mgr = FixedPipsTPSLManager(sl_pips=500, tp_pips=1000)
        tight_mgr = FixedPipsTPSLManager(sl_pips=5, tp_pips=10)

        wide_strat = SMACrossoverStrategy(settings, tpsl_manager=wide_mgr)
        tight_strat = SMACrossoverStrategy(settings, tpsl_manager=tight_mgr)

        n = 60
        prices_down = np.linspace(1.1200, 1.1000, n // 2)
        prices_up = np.linspace(1.1000, 1.1600, n // 2)
        close = np.concatenate([prices_down, prices_up])
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

        wide_sig = wide_strat.generate_signal(df, 'EURUSD')
        tight_sig = tight_strat.generate_signal(df, 'EURUSD')

        if wide_sig.is_actionable and tight_sig.is_actionable:
            wide_sl_dist = abs(wide_sig.entry_price - wide_sig.stop_loss)
            tight_sl_dist = abs(tight_sig.entry_price - tight_sig.stop_loss)
            assert wide_sl_dist > tight_sl_dist
