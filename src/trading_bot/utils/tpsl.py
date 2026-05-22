"""Take-profit / stop-loss manager base class and implementations."""

from abc import ABC, abstractmethod

from trading_bot.strategies.base_strategy import SignalType
from trading_bot.utils.pip import get_pip_size


class BaseTPSLManager(ABC):
    """Abstract interface for stop-loss / take-profit calculation.

    Subclasses implement :meth:`calculate_sl` and :meth:`calculate_tp`
    independently, so callers can request either leg without computing both.
    """

    @abstractmethod
    def calculate_sl(
        self,
        signal_type: SignalType,
        entry_price: float,
        symbol: str,
    ) -> float:
        """Return the stop-loss price for a trade.

        Args:
            signal_type: Trade direction. ``HOLD`` must return ``0.0``.
            entry_price: Confirmed fill price to measure from.
            symbol: Instrument symbol (used for pip-size resolution or
                contract-specific scaling).

        Returns:
            Absolute stop-loss price level, or ``0.0`` for
            :attr:`~SignalType.HOLD`.
        """

    @abstractmethod
    def calculate_tp(
        self,
        signal_type: SignalType,
        entry_price: float,
        symbol: str,
    ) -> float:
        """Return the take-profit price for a trade.

        Args:
            signal_type: Trade direction. ``HOLD`` must return ``0.0``.
            entry_price: Confirmed fill price to measure from.
            symbol: Instrument symbol (used for pip-size resolution or
                contract-specific scaling).

        Returns:
            Absolute take-profit price level, or ``0.0`` for
            :attr:`~SignalType.HOLD`.
        """


class FixedPipsTPSLManager(BaseTPSLManager):
    """Places SL and TP a fixed number of pips away from the entry price.

    Pip size is resolved per-instrument via
    :func:`~trading_bot.utils.pip.get_pip_size`, so the same pip counts
    work correctly across forex, metals, and crypto pairs.

    Args:
        sl_pips: Stop-loss distance in pips (must be > 0).
        tp_pips: Take-profit distance in pips (must be > 0).

    Examples:
        >>> mgr = FixedPipsTPSLManager(sl_pips=30, tp_pips=60)
        >>> mgr.calculate_sl(SignalType.BUY, 1.10000, 'EURUSD')
        1.097
        >>> mgr.calculate_tp(SignalType.BUY, 1.10000, 'EURUSD')
        1.106
    """

    def __init__(self, sl_pips: float, tp_pips: float) -> None:
        if sl_pips <= 0:
            raise ValueError(f'sl_pips must be positive, got {sl_pips}')
        if tp_pips <= 0:
            raise ValueError(f'tp_pips must be positive, got {tp_pips}')
        self._sl_pips = sl_pips
        self._tp_pips = tp_pips

    @property
    def sl_pips(self) -> float:
        """Configured stop-loss distance in pips."""
        return self._sl_pips

    @property
    def tp_pips(self) -> float:
        """Configured take-profit distance in pips."""
        return self._tp_pips

    def calculate_sl(
        self,
        signal_type: SignalType,
        entry_price: float,
        symbol: str,
    ) -> float:
        """Compute the fixed-pip stop-loss price.

        Args:
            signal_type: ``BUY`` places SL below entry; ``SELL`` places it
                above. ``HOLD`` returns ``0.0`` immediately.
            entry_price: Price to anchor the distance to.
            symbol: Used to look up the correct pip size.

        Returns:
            Absolute stop-loss price level.
        """
        if signal_type == SignalType.HOLD:
            return 0.0
        pip = get_pip_size(symbol)
        if signal_type == SignalType.BUY:
            return entry_price - self._sl_pips * pip
        # SELL
        return entry_price + self._sl_pips * pip

    def calculate_tp(
        self,
        signal_type: SignalType,
        entry_price: float,
        symbol: str,
    ) -> float:
        """Compute the fixed-pip take-profit price.

        Args:
            signal_type: ``BUY`` places TP above entry; ``SELL`` places it
                below. ``HOLD`` returns ``0.0`` immediately.
            entry_price: Price to anchor the distance to.
            symbol: Used to look up the correct pip size.

        Returns:
            Absolute take-profit price level.
        """
        if signal_type == SignalType.HOLD:
            return 0.0
        pip = get_pip_size(symbol)
        if signal_type == SignalType.BUY:
            return entry_price + self._tp_pips * pip
        # SELL
        return entry_price - self._tp_pips * pip
