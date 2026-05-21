"""Abstract base class and signal dataclass for trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd


class SignalType(StrEnum):
    """Possible signal directions."""

    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'


@dataclass
class Signal:
    """Trading signal emitted by a strategy.

    Attributes:
        signal_type: Direction of the trade recommendation.
        symbol: Instrument the signal applies to.
        entry_price: Suggested entry price.
        stop_loss: Suggested stop-loss price (0 = not set).
        take_profit: Suggested take-profit price (0 = not set).
        strength: Normalised signal strength in [0.0, 1.0].
        reason: Human-readable explanation of why the signal fired.
        metadata: Arbitrary key-value pairs from the strategy.
    """

    signal_type: SignalType
    symbol: str
    entry_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strength: float = 1.0
    reason: str = ''
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Return ``True`` if the signal is BUY or SELL."""
        return self.signal_type != SignalType.HOLD

    def __str__(self) -> str:
        return (
            f'[{self.signal_type.value}] {self.symbol} @ {self.entry_price:.5f} '
            f'| SL={self.stop_loss:.5f} TP={self.take_profit:.5f} '
            f'| strength={self.strength:.2f} | {self.reason}'
        )


class BaseStrategy(ABC):
    """Interface for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g. ``'sma_crossover'``)."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Analyse *df* and produce a :class:`Signal`.

        Args:
            df: OHLCV DataFrame (columns: ``open high low close volume``).
                Must have at least as many rows as the strategy requires.
            symbol: Instrument the data belongs to.

        Returns:
            A :class:`Signal` (may be :attr:`SignalType.HOLD`).
        """

    @abstractmethod
    def required_bars(self) -> int:
        """Minimum number of bars required to generate a valid signal."""
