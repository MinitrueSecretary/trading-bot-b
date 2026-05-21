"""Tests for the paper trading executor."""

import pytest

from trading_bot.execution.base_executor import OrderStatus
from trading_bot.execution.paper_executor import PaperExecutor
from trading_bot.strategies.base_strategy import Signal, SignalType


@pytest.fixture
def executor() -> PaperExecutor:
    """Return a fresh PaperExecutor with $10 000 balance."""
    return PaperExecutor(initial_balance=10_000.0)


@pytest.fixture
def buy_signal() -> Signal:
    """Return a sample BUY signal."""
    return Signal(
        signal_type=SignalType.BUY,
        symbol='EURUSD',
        entry_price=1.10000,
        stop_loss=1.09700,
        take_profit=1.10600,
        reason='Test BUY',
    )


@pytest.fixture
def sell_signal() -> Signal:
    """Return a sample SELL signal."""
    return Signal(
        signal_type=SignalType.SELL,
        symbol='EURUSD',
        entry_price=1.10000,
        stop_loss=1.10300,
        take_profit=1.09400,
        reason='Test SELL',
    )


class TestPaperExecutor:
    """Tests for :class:`PaperExecutor`."""

    def test_initial_balance(self, executor: PaperExecutor) -> None:
        """Initial balance should be as set."""
        assert executor.balance == 10_000.0

    def test_place_buy_order(self, executor: PaperExecutor, buy_signal: Signal) -> None:
        """Placing a BUY order should succeed and return FILLED status."""
        result = executor.place_order(buy_signal, volume=0.1)
        assert result.success is True
        assert result.status == OrderStatus.FILLED
        assert result.signal_type == SignalType.BUY
        assert len(executor.open_positions) == 1

    def test_place_sell_order(self, executor: PaperExecutor, sell_signal: Signal) -> None:
        """Placing a SELL order should succeed."""
        result = executor.place_order(sell_signal, volume=0.1)
        assert result.success is True
        assert result.status == OrderStatus.FILLED

    def test_hold_signal_not_filled(self, executor: PaperExecutor) -> None:
        """HOLD signal should be rejected."""
        hold = Signal(SignalType.HOLD, 'EURUSD', 1.1)
        result = executor.place_order(hold, volume=0.1)
        assert result.success is False
        assert result.status == OrderStatus.REJECTED

    def test_close_position(self, executor: PaperExecutor, buy_signal: Signal) -> None:
        """Closing an open position should remove it from open_positions."""
        result = executor.place_order(buy_signal, volume=0.1)
        close = executor.close_position(result.order_id, 'EURUSD')
        assert close.success is True
        assert len(executor.open_positions) == 0

    def test_close_nonexistent_position(self, executor: PaperExecutor) -> None:
        """Closing a non-existent position should return failure."""
        result = executor.close_position('nonexistent', 'EURUSD')
        assert result.success is False

    def test_tp_hit_closes_position(self, executor: PaperExecutor, buy_signal: Signal) -> None:
        """When price hits TP, the position should be auto-closed."""
        executor.place_order(buy_signal, volume=0.1)
        closed = executor.update_positions('EURUSD', 1.10650)  # above TP=1.10600
        assert len(closed) == 1
        assert 'TP' in closed[0].message
        assert len(executor.open_positions) == 0

    def test_sl_hit_closes_position(self, executor: PaperExecutor, buy_signal: Signal) -> None:
        """When price hits SL, the position should be auto-closed with a loss."""
        executor.place_order(buy_signal, volume=0.1)
        closed = executor.update_positions('EURUSD', 1.09650)  # below SL=1.09700
        assert len(closed) == 1
        assert 'SL' in closed[0].message

    def test_get_stats_empty(self, executor: PaperExecutor) -> None:
        """Stats on no trades should return zeros."""
        stats = executor.get_stats()
        assert stats['total_trades'] == 0
        assert stats['win_rate'] == 0.0

    def test_get_trade_history_after_close(
        self, executor: PaperExecutor, buy_signal: Signal
    ) -> None:
        """Closed trades should appear in trade history."""
        result = executor.place_order(buy_signal, volume=0.1)
        executor.close_position(result.order_id, 'EURUSD')
        history = executor.get_trade_history()
        assert len(history) == 1
        assert 'pnl' in history.columns
