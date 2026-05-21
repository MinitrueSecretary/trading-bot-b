"""Tests for the backtesting engine."""

import pandas as pd
import pytest

from trading_bot.backtesting import Backtester, BacktestResult
from trading_bot.config import Settings
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy


@pytest.fixture
def backtester(settings: Settings) -> Backtester:
    """Return a Backtester with SMA strategy."""
    strat = SMACrossoverStrategy(settings)
    return Backtester(strat, initial_balance=10_000.0, lot_size=0.1)


class TestBacktester:
    """Tests for :class:`Backtester`."""

    def test_run_returns_result(self, backtester: Backtester, ohlcv_df: pd.DataFrame) -> None:
        """run() should return a BacktestResult."""
        result = backtester.run(ohlcv_df, 'EURUSD')
        assert isinstance(result, BacktestResult)

    def test_result_symbol_and_strategy(
        self, backtester: Backtester, ohlcv_df: pd.DataFrame
    ) -> None:
        """Result should capture symbol and strategy name."""
        result = backtester.run(ohlcv_df, 'EURUSD')
        assert result.symbol == 'EURUSD'
        assert result.strategy_name == 'sma_crossover'

    def test_trade_count_non_negative(
        self, backtester: Backtester, ohlcv_df: pd.DataFrame
    ) -> None:
        """Total trades should be non-negative."""
        result = backtester.run(ohlcv_df, 'EURUSD')
        assert result.total_trades >= 0

    def test_win_rate_in_range(self, backtester: Backtester, ohlcv_df: pd.DataFrame) -> None:
        """Win rate should be between 0 and 100."""
        result = backtester.run(ohlcv_df, 'EURUSD')
        assert 0.0 <= result.win_rate <= 100.0

    def test_equity_curve_starts_at_initial_balance(
        self, backtester: Backtester, ohlcv_df: pd.DataFrame
    ) -> None:
        """Equity curve first value should equal initial balance."""
        result = backtester.run(ohlcv_df, 'EURUSD')
        assert result.equity_curve.iloc[0] == pytest.approx(10_000.0, rel=1e-3)

    def test_max_drawdown_non_negative(
        self, backtester: Backtester, ohlcv_df: pd.DataFrame
    ) -> None:
        """Max drawdown should be non-negative."""
        result = backtester.run(ohlcv_df, 'EURUSD')
        assert result.max_drawdown_pct >= 0.0

    def test_summary_string(self, backtester: Backtester, ohlcv_df: pd.DataFrame) -> None:
        """summary() should return a non-empty string."""
        result = backtester.run(ohlcv_df, 'EURUSD')
        summary = result.summary()
        assert 'EURUSD' in summary
        assert 'sma_crossover' in summary
