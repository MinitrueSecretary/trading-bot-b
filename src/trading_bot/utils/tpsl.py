"""Take-profit / stop-loss manager base class and implementations."""

from abc import ABC, abstractmethod

import pandas as pd
import pandas_ta as ta  # type: ignore[import]

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


class BollingerBandsTPSLManager(BaseTPSLManager):
    """Places SL and TP at the Bollinger Band levels closest to the current price.

    For a **BUY** trade the lower band becomes the stop-loss and the upper band
    becomes the take-profit.  For a **SELL** trade the mapping is reversed.

    Because Bollinger Bands require historical OHLCV data, callers **must** call
    :meth:`set_data` with the current price DataFrame before invoking
    :meth:`calculate_sl` or :meth:`calculate_tp`.

    Args:
        period: Look-back window for the middle SMA and band calculation.
            Defaults to ``20``.
        std_dev: Number of standard deviations for the band width. Defaults
            to ``2.0``.
        min_sl_pips: Optional minimum stop-loss distance in pips.  When the
            band-derived SL would be *closer* than this floor, the floor is
            used instead.  Defaults to ``0.0`` (disabled).

    Raises:
        ValueError: If ``period`` < 2, ``std_dev`` <= 0, or
            ``min_sl_pips`` < 0.
        RuntimeError: If :meth:`calculate_sl` or :meth:`calculate_tp` is
            called before :meth:`set_data`.

    Examples:
        >>> mgr = BollingerBandsTPSLManager(period=20, std_dev=2.0)
        >>> mgr.set_data(df)  # df is an OHLCV DataFrame
        >>> mgr.calculate_sl(SignalType.BUY, entry_price, 'EURUSD')
        1.09712  # lower band value
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        min_sl_pips: float = 0.0,
    ) -> None:
        if period < 2:
            raise ValueError(f'period must be >= 2, got {period}')
        if std_dev <= 0:
            raise ValueError(f'std_dev must be positive, got {std_dev}')
        if min_sl_pips < 0:
            raise ValueError(f'min_sl_pips must be >= 0, got {min_sl_pips}')
        self._period = period
        self._std_dev = std_dev
        self._min_sl_pips = min_sl_pips
        self._df: pd.DataFrame | None = None

    @property
    def period(self) -> int:
        """Configured Bollinger Band look-back period."""
        return self._period

    @property
    def std_dev(self) -> float:
        """Configured number of standard deviations for band width."""
        return self._std_dev

    @property
    def min_sl_pips(self) -> float:
        """Minimum stop-loss distance floor in pips (0.0 = disabled)."""
        return self._min_sl_pips

    def set_data(self, df: pd.DataFrame) -> None:
        """Supply the OHLCV DataFrame used for band calculation.

        Call this once per
        :meth:`~trading_bot.strategies.base_strategy.BaseStrategy.generate_signal`
        invocation, before calling :meth:`calculate_sl` or
        :meth:`calculate_tp`.

        Args:
            df: OHLCV DataFrame with at least a ``close`` column and enough
                rows to compute the band (``>= period``).
        """
        self._df = df

    def _get_bands(self) -> tuple[float, float]:
        """Compute and return the current ``(lower_band, upper_band)`` values.

        Returns:
            A ``(lower, upper)`` tuple of the most-recent Bollinger Band levels.

        Raises:
            RuntimeError: If no DataFrame has been set via :meth:`set_data`.
            ValueError: If the DataFrame has too few rows to compute bands.
        """
        if self._df is None:
            raise RuntimeError(
                'BollingerBandsTPSLManager.set_data() must be called before '
                'calculate_sl() / calculate_tp().'
            )
        if len(self._df) < self._period:
            raise ValueError(
                f'DataFrame has {len(self._df)} rows but period={self._period} '
                f'requires at least {self._period} rows.'
            )
        bbands = ta.bbands(
            self._df['close'],
            length=self._period,
            lower_std=self._std_dev,
            upper_std=self._std_dev,
        )
        # pandas-ta column names vary by version; detect by prefix to stay robust.
        cols = bbands.columns.tolist()
        lower_col = next((c for c in cols if c.startswith('BBL_')), None)
        upper_col = next((c for c in cols if c.startswith('BBU_')), None)
        if lower_col is None or upper_col is None:
            raise ValueError(
                f'Unexpected pandas-ta bbands columns: {cols}. '
                'Expected columns starting with BBL_ and BBU_.'
            )
        lower = float(bbands[lower_col].iloc[-1])
        upper = float(bbands[upper_col].iloc[-1])
        return lower, upper

    def calculate_sl(
        self,
        signal_type: SignalType,
        entry_price: float,
        symbol: str,
    ) -> float:
        """Compute the Bollinger Band stop-loss price.

        * **BUY** → SL at the lower band (floored by ``min_sl_pips`` if set).
        * **SELL** → SL at the upper band (ceilinged by ``min_sl_pips`` if set).
        * **HOLD** → ``0.0``.

        Args:
            signal_type: Trade direction.
            entry_price: Confirmed fill price.
            symbol: Instrument symbol for pip-size resolution.

        Returns:
            Absolute stop-loss price level, or ``0.0`` for HOLD.
        """
        if signal_type == SignalType.HOLD:
            return 0.0
        lower, upper = self._get_bands()
        pip = get_pip_size(symbol)
        min_dist = self._min_sl_pips * pip
        if signal_type == SignalType.BUY:
            sl = lower
            if min_dist > 0:
                floor = entry_price - min_dist
                sl = min(sl, floor)
            return sl
        # SELL
        sl = upper
        if min_dist > 0:
            ceiling = entry_price + min_dist
            sl = max(sl, ceiling)
        return sl

    def calculate_tp(
        self,
        signal_type: SignalType,
        entry_price: float,
        symbol: str,
    ) -> float:
        """Compute the Bollinger Band take-profit price.

        * **BUY** → TP at the upper band.
        * **SELL** → TP at the lower band.
        * **HOLD** → ``0.0``.

        Args:
            signal_type: Trade direction.
            entry_price: Confirmed fill price.
            symbol: Instrument symbol (unused, kept for interface parity).

        Returns:
            Absolute take-profit price level, or ``0.0`` for HOLD.
        """
        if signal_type == SignalType.HOLD:
            return 0.0
        lower, upper = self._get_bands()
        if signal_type == SignalType.BUY:
            return upper
        # SELL
        return lower
