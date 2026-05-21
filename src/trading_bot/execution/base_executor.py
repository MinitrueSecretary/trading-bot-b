"""Abstract base class for order executors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from trading_bot.strategies.base_strategy import Signal, SignalType


class OrderStatus(StrEnum):
    """Order lifecycle status."""

    PENDING = 'PENDING'
    FILLED = 'FILLED'
    REJECTED = 'REJECTED'
    CANCELLED = 'CANCELLED'


@dataclass
class OrderResult:
    """Result returned after attempting to place an order.

    Attributes:
        success: Whether the order was accepted.
        order_id: Broker-assigned order ID (or paper-trading UUID).
        symbol: Instrument traded.
        signal_type: BUY or SELL.
        price: Fill price (or 0 if rejected).
        volume: Lot size traded.
        stop_loss: Stop-loss price.
        take_profit: Take-profit price.
        status: Current order status.
        message: Human-readable result message.
        timestamp: Time the order result was created.
    """

    success: bool
    order_id: str
    symbol: str
    signal_type: SignalType
    price: float
    volume: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    message: str = ''
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __str__(self) -> str:
        return (
            f'[{self.status.value}] #{self.order_id} {self.signal_type.value} '
            f'{self.volume} {self.symbol} @ {self.price:.5f} '
            f'SL={self.stop_loss:.5f} TP={self.take_profit:.5f} — {self.message}'
        )


class BaseExecutor(ABC):
    """Interface that every order executor must implement."""

    @abstractmethod
    def place_order(self, signal: Signal, volume: float) -> OrderResult:
        """Submit an order based on *signal*.

        Args:
            signal: The trading signal to act on.
            volume: Lot size to trade.

        Returns:
            An :class:`OrderResult` describing the outcome.
        """

    @abstractmethod
    def close_position(self, order_id: str, symbol: str) -> OrderResult:
        """Close an existing position.

        Args:
            order_id: The order/position identifier.
            symbol: Instrument symbol.

        Returns:
            An :class:`OrderResult` for the close operation.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable executor name (e.g. ``'PaperTrading'``)."""
