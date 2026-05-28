"""Tests for CompositeStrategy and load_composite_strategy."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import Signal, SignalType
from trading_bot.strategies.composite import (
    CombineMode,
    CompositeStrategy,
    StrategyEntry,
    load_composite_strategy,
)
from trading_bot.strategies.macd_strategy import MACDStrategy
from trading_bot.strategies.rsi_strategy import RSIStrategy
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_df(n: int = 300) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    close = 1.1 + np.cumsum(rng.normal(0, 0.0002, n))
    idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    df = pd.DataFrame(
        {
            'open': close - 0.0001,
            'high': close + 0.0005,
            'low': close - 0.0005,
            'close': close,
            'volume': np.ones(n) * 1000.0,
        },
        index=idx,
    )
    df['high'] = df[['high', 'close', 'open']].max(axis=1)
    df['low'] = df[['low', 'close', 'open']].min(axis=1)
    return df


class _FixedSignalStrategy(SMACrossoverStrategy):
    """Stub that always returns a preset signal, for deterministic tests."""

    def __init__(self, settings: Settings, fixed_signal: Signal) -> None:
        super().__init__(settings)
        self._fixed = fixed_signal

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        return self._fixed


def _buy(settings: Settings, price: float = 1.1, strength: float = 0.8) -> _FixedSignalStrategy:
    sig = Signal(SignalType.BUY, 'EURUSD', price, stop_loss=price - 0.003,
                 take_profit=price + 0.006, strength=strength)
    return _FixedSignalStrategy(settings, sig)


def _sell(settings: Settings, price: float = 1.1, strength: float = 0.8) -> _FixedSignalStrategy:
    sig = Signal(SignalType.SELL, 'EURUSD', price, stop_loss=price + 0.003,
                 take_profit=price - 0.006, strength=strength)
    return _FixedSignalStrategy(settings, sig)


def _hold(settings: Settings) -> _FixedSignalStrategy:
    sig = Signal(SignalType.HOLD, 'EURUSD', 1.1)
    return _FixedSignalStrategy(settings, sig)


# ── StrategyEntry ─────────────────────────────────────────────────────────────


class TestStrategyEntry:
    """Tests for :class:`StrategyEntry.applies_to`."""

    def test_wildcard_applies_to_any_symbol(self, settings: Settings) -> None:
        e = StrategyEntry(strategy=SMACrossoverStrategy(settings), symbols=['*'])
        assert e.applies_to('EURUSD')
        assert e.applies_to('BTCUSD')

    def test_specific_symbols_match_case_insensitive(self, settings: Settings) -> None:
        e = StrategyEntry(strategy=SMACrossoverStrategy(settings), symbols=['eurusd', 'GBPUSD'])
        assert e.applies_to('EURUSD')
        assert e.applies_to('gbpusd')
        assert not e.applies_to('USDJPY')

    def test_does_not_apply_to_unlisted_symbol(self, settings: Settings) -> None:
        e = StrategyEntry(strategy=SMACrossoverStrategy(settings), symbols=['EURUSD'])
        assert not e.applies_to('USDJPY')


# ── CompositeStrategy construction ────────────────────────────────────────────


class TestCompositeConstruction:
    """Tests for basic CompositeStrategy properties."""

    def test_empty_entries_raises(self, settings: Settings) -> None:
        with pytest.raises(ValueError, match='at least one'):
            CompositeStrategy(entries=[])

    def test_name_is_composite(self, settings: Settings) -> None:
        e = StrategyEntry(SMACrossoverStrategy(settings), symbols=['*'])
        assert CompositeStrategy([e]).name == 'composite'

    def test_required_bars_is_max(self, settings: Settings) -> None:
        sma = SMACrossoverStrategy(settings)
        rsi = RSIStrategy(settings)
        c = CompositeStrategy([
            StrategyEntry(sma, symbols=['*']),
            StrategyEntry(rsi, symbols=['*']),
        ])
        assert c.required_bars() == max(sma.required_bars(), rsi.required_bars())

    def test_mode_property(self, settings: Settings) -> None:
        e = StrategyEntry(SMACrossoverStrategy(settings), symbols=['*'])
        c = CompositeStrategy([e], mode=CombineMode.WEIGHTED_VOTE)
        assert c.mode == CombineMode.WEIGHTED_VOTE

    def test_entries_property_returns_copy(self, settings: Settings) -> None:
        e = StrategyEntry(SMACrossoverStrategy(settings), symbols=['*'])
        c = CompositeStrategy([e])
        assert c.entries is not c.entries  # different list each call


# ── No applicable strategy ────────────────────────────────────────────────────


class TestNoApplicableStrategy:
    """HOLD is returned when no entry is scoped to the requested symbol."""

    def test_hold_when_symbol_not_in_any_entry(self, settings: Settings) -> None:
        e = StrategyEntry(_buy(settings), symbols=['GBPUSD'])
        c = CompositeStrategy([e])
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.HOLD
        assert 'No strategy configured' in sig.reason


# ── FIRST_SIGNAL mode ─────────────────────────────────────────────────────────


class TestFirstSignalMode:
    """Tests for CombineMode.FIRST_SIGNAL."""

    def test_first_buy_wins(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*']),
                StrategyEntry(_sell(settings), symbols=['*']),
            ],
            mode=CombineMode.FIRST_SIGNAL,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.BUY

    def test_hold_skipped_in_favour_of_second(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_hold(settings), symbols=['*']),
                StrategyEntry(_sell(settings), symbols=['*']),
            ],
            mode=CombineMode.FIRST_SIGNAL,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.SELL

    def test_all_hold_returns_hold(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_hold(settings), symbols=['*']),
                StrategyEntry(_hold(settings), symbols=['*']),
            ],
            mode=CombineMode.FIRST_SIGNAL,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.HOLD


# ── ALL_AGREE mode ────────────────────────────────────────────────────────────


class TestAllAgreeMode:
    """Tests for CombineMode.ALL_AGREE."""

    def test_all_buy_produces_buy(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings, strength=0.5), symbols=['*']),
                StrategyEntry(_buy(settings, strength=0.9), symbols=['*']),
            ],
            mode=CombineMode.ALL_AGREE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.BUY

    def test_disagreement_produces_hold(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*']),
                StrategyEntry(_sell(settings), symbols=['*']),
            ],
            mode=CombineMode.ALL_AGREE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.HOLD

    def test_hold_counts_as_disagree(self, settings: Settings) -> None:
        """ALL_AGREE requires every entry to be actionable in the same direction."""
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*']),
                StrategyEntry(_hold(settings), symbols=['*']),
            ],
            mode=CombineMode.ALL_AGREE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.HOLD

    def test_picks_highest_strength(self, settings: Settings) -> None:
        weak = _buy(settings, strength=0.3)
        strong = _buy(settings, strength=0.95)
        c = CompositeStrategy(
            [StrategyEntry(weak, symbols=['*']), StrategyEntry(strong, symbols=['*'])],
            mode=CombineMode.ALL_AGREE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.BUY
        assert sig.strength == pytest.approx(0.95)


# ── MAJORITY_VOTE mode ────────────────────────────────────────────────────────


class TestMajorityVoteMode:
    """Tests for CombineMode.MAJORITY_VOTE (default)."""

    def test_two_buy_one_sell_produces_buy(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*']),
                StrategyEntry(_buy(settings), symbols=['*']),
                StrategyEntry(_sell(settings), symbols=['*']),
            ],
            mode=CombineMode.MAJORITY_VOTE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.BUY

    def test_tie_produces_hold(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*']),
                StrategyEntry(_sell(settings), symbols=['*']),
            ],
            mode=CombineMode.MAJORITY_VOTE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.HOLD
        assert 'Tied' in sig.reason

    def test_composite_strength_is_fraction_of_total(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*'], weight=2.0),
                StrategyEntry(_sell(settings), symbols=['*'], weight=1.0),
            ],
            mode=CombineMode.MAJORITY_VOTE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.BUY
        assert sig.strength == pytest.approx(2.0 / 3.0)

    def test_metadata_contains_votes(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*']),
                StrategyEntry(_sell(settings), symbols=['*']),
                StrategyEntry(_buy(settings), symbols=['*']),
            ],
            mode=CombineMode.MAJORITY_VOTE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert 'votes' in sig.metadata
        assert 'buy_score' in sig.metadata
        assert 'sell_score' in sig.metadata


# ── WEIGHTED_VOTE mode ────────────────────────────────────────────────────────


class TestWeightedVoteMode:
    """Tests for CombineMode.WEIGHTED_VOTE."""

    def test_higher_weight_wins(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_sell(settings), symbols=['*'], weight=1.0),
                StrategyEntry(_buy(settings), symbols=['*'], weight=3.0),
            ],
            mode=CombineMode.WEIGHTED_VOTE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.BUY

    def test_equal_weight_tie_is_hold(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['*'], weight=1.5),
                StrategyEntry(_sell(settings), symbols=['*'], weight=1.5),
            ],
            mode=CombineMode.WEIGHTED_VOTE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.HOLD


# ── Symbol scoping ────────────────────────────────────────────────────────────


class TestSymbolScoping:
    """Only entries that apply to the requested symbol contribute."""

    def test_only_scoped_entry_fires(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [
                StrategyEntry(_buy(settings), symbols=['EURUSD']),
                StrategyEntry(_sell(settings), symbols=['GBPUSD']),
            ],
            mode=CombineMode.MAJORITY_VOTE,
        )
        sig = c.generate_signal(_make_df(), 'EURUSD')
        assert sig.signal_type == SignalType.BUY

    def test_wildcard_entry_applies_to_all(self, settings: Settings) -> None:
        c = CompositeStrategy(
            [StrategyEntry(_sell(settings), symbols=['*'])],
            mode=CombineMode.MAJORITY_VOTE,
        )
        for sym in ('EURUSD', 'USDJPY', 'BTCUSD'):
            sig = c.generate_signal(_make_df(), sym)
            assert sig.signal_type == SignalType.SELL


# ── YAML config loading ───────────────────────────────────────────────────────


class TestLoadCompositeStrategy:
    """Tests for :func:`load_composite_strategy`."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / 'cfg.yaml'
        p.write_text(textwrap.dedent(content))
        return p

    def test_basic_yaml_loads(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            mode: majority_vote
            strategies:
              - strategy: sma_crossover
                symbols: ["*"]
                weight: 1.0
                params:
                  sma_fast: 5
                  sma_slow: 20
        """)
        composite = load_composite_strategy(cfg, settings)
        assert isinstance(composite, CompositeStrategy)
        assert composite.mode == CombineMode.MAJORITY_VOTE
        assert len(composite.entries) == 1

    def test_disabled_strategy_is_skipped(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            mode: first_signal
            strategies:
              - strategy: sma_crossover
                enabled: false
                symbols: ["*"]
              - strategy: macd
                enabled: true
                symbols: ["*"]
        """)
        composite = load_composite_strategy(cfg, settings)
        assert len(composite.entries) == 1
        assert composite.entries[0].strategy.name == 'macd'

    def test_params_override_settings(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            mode: majority_vote
            strategies:
              - strategy: sma_crossover
                symbols: ["*"]
                params:
                  sma_fast: 7
                  sma_slow: 21
        """)
        composite = load_composite_strategy(cfg, settings)
        # The strategy instance should have the overridden SMA params
        strat = composite.entries[0].strategy
        assert isinstance(strat, SMACrossoverStrategy)
        assert strat._fast == 7  # noqa: SLF001
        assert strat._slow == 21  # noqa: SLF001

    def test_symbol_scoping_from_yaml(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            mode: majority_vote
            strategies:
              - strategy: rsi
                symbols: ["EURUSD", "GBPUSD"]
        """)
        composite = load_composite_strategy(cfg, settings)
        entry = composite.entries[0]
        assert entry.applies_to('EURUSD')
        assert entry.applies_to('GBPUSD')
        assert not entry.applies_to('USDJPY')

    def test_weight_from_yaml(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            mode: weighted_vote
            strategies:
              - strategy: sma_crossover
                symbols: ["*"]
                weight: 3.5
        """)
        composite = load_composite_strategy(cfg, settings)
        assert composite.entries[0].weight == pytest.approx(3.5)

    def test_unknown_strategy_raises(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            strategies:
              - strategy: nonexistent_strategy
                symbols: ["*"]
        """)
        with pytest.raises(ValueError, match='nonexistent_strategy'):
            load_composite_strategy(cfg, settings)

    def test_all_disabled_raises(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            strategies:
              - strategy: sma_crossover
                enabled: false
                symbols: ["*"]
        """)
        with pytest.raises(ValueError, match='No enabled strategies'):
            load_composite_strategy(cfg, settings)

    def test_missing_file_raises(self, settings: Settings, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_composite_strategy(tmp_path / 'missing.yaml', settings)

    def test_all_combine_modes_accepted(self, settings: Settings, tmp_path: Path) -> None:
        for mode in CombineMode:
            cfg = self._write_yaml(tmp_path, f"""
                mode: {mode.value}
                strategies:
                  - strategy: sma_crossover
                    symbols: ["*"]
            """)
            composite = load_composite_strategy(cfg, settings)
            assert composite.mode == mode

    def test_multiple_strategies_in_yaml(self, settings: Settings, tmp_path: Path) -> None:
        cfg = self._write_yaml(tmp_path, """
            mode: weighted_vote
            strategies:
              - strategy: sma_crossover
                symbols: ["EURUSD"]
                weight: 1.0
              - strategy: macd
                symbols: ["EURUSD"]
                weight: 2.0
              - strategy: rsi
                symbols: ["GBPUSD"]
                weight: 1.5
        """)
        composite = load_composite_strategy(cfg, settings)
        assert len(composite.entries) == 3
        assert composite.mode == CombineMode.WEIGHTED_VOTE

    def test_generate_signal_runs_without_error(
        self, settings: Settings, tmp_path: Path, ohlcv_df: pd.DataFrame
    ) -> None:
        """Integration: load from YAML and generate a signal end-to-end."""
        cfg = self._write_yaml(tmp_path, """
            mode: majority_vote
            strategies:
              - strategy: sma_crossover
                symbols: ["*"]
              - strategy: macd
                symbols: ["*"]
        """)
        composite = load_composite_strategy(cfg, settings)
        sig = composite.generate_signal(ohlcv_df, 'EURUSD')
        assert sig.signal_type in {SignalType.BUY, SignalType.SELL, SignalType.HOLD}
