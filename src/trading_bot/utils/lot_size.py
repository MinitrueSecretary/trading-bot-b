"""Lot size manager base class and implementations."""

import logging
from abc import ABC, abstractmethod

from trading_bot.utils.pip import get_contract_size

logger = logging.getLogger(__name__)


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


class FixedRiskLotSizeManager(BaseLotSizeManager):
    """Sizes positions so a stop-loss hit loses a fixed dollar amount.

    The lot size is calculated by working backwards from the desired dollar
    risk and the distance to the stop-loss:

        lot_size = risk_amount / (sl_distance * contract_size)

    This matches the P&L formula used throughout the codebase
    ``pnl = price_change * lot_size * contract_size``, so a stop-loss hit
    will lose approximately *risk_amount* dollars.

    The contract size is looked up automatically from :func:`get_contract_size`
    using the *symbol* passed to :meth:`get_lot_size`, so the correct value is
    used for every instrument (forex, metals, crypto, …).

    When no stop-loss price is provided, or the computed lot would fall below
    *min_lot*, the manager falls back to *min_lot* and logs a warning.

    Args:
        risk_amount: Fixed dollar amount to risk per trade (must be > 0).
        min_lot: Minimum returnable lot size (default ``0.01``, the smallest
            standard MT5 lot step). Acts as a floor (must be > 0).

    Raises:
        ValueError: If *risk_amount* or *min_lot* are not positive.

    Examples:
        >>> mgr = FixedRiskLotSizeManager(risk_amount=100.0)
        >>> # EURUSD BUY with SL 30 pips away (0.003 price units)
        >>> mgr.get_lot_size(10_000, 1.10000, 'EURUSD', stop_loss_price=1.09700)
        0.33
    """

    def __init__(
        self,
        risk_amount: float,
        min_lot: float = 0.01,
    ) -> None:
        if risk_amount <= 0:
            raise ValueError(f'risk_amount must be positive, got {risk_amount}')
        if min_lot <= 0:
            raise ValueError(f'min_lot must be positive, got {min_lot}')
        self._risk_amount = risk_amount
        self._min_lot = min_lot

    @property
    def risk_amount(self) -> float:
        """Fixed dollar amount at risk per trade."""
        return self._risk_amount

    @property
    def min_lot(self) -> float:
        """Minimum lot size returned when SL is absent or the computed lot is too small."""
        return self._min_lot

    def get_lot_size(
        self,
        balance: float,
        entry_price: float,
        symbol: str,
        stop_loss_price: float = 0.0,
    ) -> float:
        """Calculate the lot size that risks exactly *risk_amount* dollars at the SL.

        Args:
            balance: Ignored (risk is fixed, not a fraction of balance).
            entry_price: Intended fill price.
            symbol: Instrument symbol used to look up the contract size via
                :func:`~trading_bot.utils.pip.get_contract_size`.
            stop_loss_price: Absolute stop-loss price level. When ``0.0`` or
                equal to *entry_price*, falls back to :attr:`min_lot`.

        Returns:
            Lot size rounded to 2 decimal places, never below :attr:`min_lot`.
        """
        sl_distance = abs(entry_price - stop_loss_price)

        if stop_loss_price == 0.0 or sl_distance == 0.0:
            logger.warning(
                'FixedRiskLotSizeManager: no valid stop-loss distance — '
                f'falling back to min_lot={self._min_lot}'
            )
            return self._min_lot

        contract_size = get_contract_size(symbol)
        raw = self._risk_amount / (sl_distance * contract_size)
        lot = round(raw, 2)

        if lot < self._min_lot:
            logger.warning(
                f'FixedRiskLotSizeManager: computed lot {lot:.4f} < min_lot={self._min_lot} — '
                'clamping up'
            )
            return self._min_lot

        return lot
