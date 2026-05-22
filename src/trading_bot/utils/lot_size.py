"""Lot size manager base class and implementations."""

from abc import ABC, abstractmethod


class BaseLotSizeManager(ABC):
    """Abstract interface for lot size calculation.

    Subclasses implement :meth:`get_lot_size` to return the appropriate
    position size given the current account state and trade parameters.
    """

    @abstractmethod
    def get_lot_size(
        self,
        balance: float,
        entry_price: float,
        symbol: str,
        stop_loss_price: float = 0.0,
    ) -> float:
        """Calculate the lot size for an upcoming trade.

        Args:
            balance: Current account balance in the account's base currency.
            entry_price: Intended fill price.
            symbol: Instrument symbol (e.g. ``'EURUSD'``, ``'BTCUSD'``).
            stop_loss_price: Absolute stop-loss price level (``0.0`` = not set).
                Risk-based implementations use the distance to size the position.

        Returns:
            Lot size as a ``float``.
        """


class FixedLotSizeManager(BaseLotSizeManager):
    """Returns a constant lot size for every trade regardless of account state.

    This is the simplest possible implementation — useful for backtesting and
    paper trading where fixed-size positions are intentional.

    Args:
        lot_size: Lot size to use for every trade (default ``0.1``).

    Raises:
        ValueError: If *lot_size* is not positive.

    Examples:
        >>> mgr = FixedLotSizeManager(lot_size=0.5)
        >>> mgr.get_lot_size(balance=10_000, entry_price=1.1, symbol='EURUSD')
        0.5
    """

    def __init__(self, lot_size: float = 0.1) -> None:
        if lot_size <= 0:
            raise ValueError(f'lot_size must be positive, got {lot_size}')
        self._lot_size = lot_size

    @property
    def lot_size(self) -> float:
        """The configured fixed lot size."""
        return self._lot_size

    def get_lot_size(
        self,
        balance: float,
        entry_price: float,
        symbol: str,
        stop_loss_price: float = 0.0,
    ) -> float:
        """Return the fixed lot size, ignoring all trade parameters.

        Args:
            balance: Ignored.
            entry_price: Ignored.
            symbol: Ignored.
            stop_loss_price: Ignored.

        Returns:
            The configured fixed lot size.
        """
        return self._lot_size
