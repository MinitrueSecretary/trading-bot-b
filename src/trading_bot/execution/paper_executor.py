"""Paper trading executor — simulates order fills without a broker."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from trading_bot.execution.base_executor import BaseExecutor, OrderResult, OrderStatus
from trading_bot.strategies.base_strategy import Signal, SignalType
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PaperPosition:
    """Represents an open paper-trade position.

    Attributes:
        order_id: Unique identifier.
        symbol: Instrument.
        signal_type: BUY or SELL.
        open_price: Price at entry.
        volume: Lot size.
        stop_loss: Stop-loss level.
        take_profit: Take-profit level.
        opened_at: UTC timestamp of entry.
        pnl: Running profit/loss in price units.
    """

    order_id: str
    symbol: str
    signal_type: SignalType
    open_price: float
    volume: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    pnl: float = 0.0


class PaperExecutor(BaseExecutor):
    """Simulates order execution in-memory without touching a broker.

    Tracks open positions and computes P&L.  Call :meth:`update_positions`
    on each new price tick to check for SL/TP hits.

    Args:
        initial_balance: Starting paper account balance.
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._positions: dict[str, PaperPosition] = {}
        self._closed_trades: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        """Return executor name."""
        return 'PaperTrading'

    @property
    def balance(self) -> float:
        """Current paper balance."""
        return self._balance

    @property
    def open_positions(self) -> list[PaperPosition]:
        """List of all currently open paper positions."""
        return list(self._positions.values())

    def place_order(self, signal: Signal, volume: float) -> OrderResult:
        """Simulate placing a market order.

        Args:
            signal: Trading signal with entry price, SL, TP.
            volume: Lot size.

        Returns:
            An :class:`OrderResult` with status FILLED (always succeeds in paper mode).
        """
        if signal.signal_type == SignalType.HOLD:
            return OrderResult(
                success=False,
                order_id='',
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                price=signal.entry_price,
                volume=volume,
                status=OrderStatus.REJECTED,
                message='HOLD signal — no order placed',
            )

        order_id = str(uuid.uuid4())[:8]
        pos = PaperPosition(
            order_id=order_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            open_price=signal.entry_price,
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        self._positions[order_id] = pos

        result = OrderResult(
            success=True,
            order_id=order_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            price=signal.entry_price,
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=OrderStatus.FILLED,
            message=f'Paper order filled @ {signal.entry_price:.5f}',
        )
        logger.info(f'[Paper] Order placed: {result}')
        return result

    def close_position(self, order_id: str, symbol: str) -> OrderResult:
        """Close a paper position by ID.

        Args:
            order_id: Position identifier.
            symbol: Instrument (used for price lookup if needed).

        Returns:
            An :class:`OrderResult` for the close.
        """
        pos = self._positions.pop(order_id, None)
        if pos is None:
            return OrderResult(
                success=False,
                order_id=order_id,
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=0.0,
                volume=0.0,
                status=OrderStatus.REJECTED,
                message=f'Position {order_id} not found',
            )

        self._balance += pos.pnl
        self._closed_trades.append(
            {
                'order_id': pos.order_id,
                'symbol': pos.symbol,
                'direction': pos.signal_type.value,
                'open_price': pos.open_price,
                'volume': pos.volume,
                'pnl': pos.pnl,
                'opened_at': pos.opened_at,
                'closed_at': datetime.now(UTC),
            }
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            signal_type=pos.signal_type,
            price=pos.open_price,
            volume=pos.volume,
            status=OrderStatus.FILLED,
            message=f'Position closed | P&L={pos.pnl:+.2f}',
        )

    def update_positions(self, symbol: str, current_price: float) -> list[OrderResult]:
        """Check open positions for SL/TP hits and settle them.

        Args:
            symbol: Instrument to check.
            current_price: Latest market price.

        Returns:
            List of :class:`OrderResult` for any positions that were closed.
        """
        closed: list[OrderResult] = []
        for order_id, pos in list(self._positions.items()):
            if pos.symbol != symbol:
                continue

            # Update running P&L (simplified: ignores spreads, commissions)
            if pos.signal_type == SignalType.BUY:
                pos.pnl = (current_price - pos.open_price) * pos.volume * 100_000
                hit_sl = current_price <= pos.stop_loss and pos.stop_loss > 0
                hit_tp = current_price >= pos.take_profit and pos.take_profit > 0
            else:
                pos.pnl = (pos.open_price - current_price) * pos.volume * 100_000
                hit_sl = current_price >= pos.stop_loss and pos.stop_loss > 0
                hit_tp = current_price <= pos.take_profit and pos.take_profit > 0

            if hit_sl or hit_tp:
                reason = 'TP hit' if hit_tp else 'SL hit'
                logger.info(f'[Paper] {reason} for {order_id} | P&L={pos.pnl:+.2f}')
                result = self.close_position(order_id, symbol)
                result.message = f'{reason} @ {current_price:.5f} | P&L={pos.pnl:+.2f}'
                closed.append(result)

        return closed

    def get_trade_history(self) -> pd.DataFrame:
        """Return all closed trades as a DataFrame.

        Returns:
            DataFrame of closed trades with P&L.
        """
        if not self._closed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self._closed_trades)

    def get_stats(self) -> dict[str, float]:
        """Calculate summary statistics from closed trades.

        Returns:
            Dict with keys: total_trades, win_rate, total_pnl, net_return_pct.
        """
        if not self._closed_trades:
            return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'net_return_pct': 0.0}

        pnls = [t['pnl'] for t in self._closed_trades]  # type: ignore[index]
        total = len(pnls)
        wins = sum(1 for p in pnls if p > 0)  # type: ignore[operator]
        total_pnl = sum(pnls)  # type: ignore[arg-type]

        return {
            'total_trades': float(total),
            'win_rate': wins / total * 100,
            'total_pnl': float(total_pnl),
            'net_return_pct': (float(total_pnl) / self._initial_balance) * 100,
        }
