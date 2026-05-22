"""SMA Crossover strategy — fast MA crosses above/below slow MA."""

import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)


class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy.

    Generates a **BUY** signal when the fast SMA crosses above the slow SMA,
    and a **SELL** signal on the opposite crossover.

    Args:
        settings: Application settings providing ``sma_fast``, ``sma_slow``,
            ``stop_loss_pips``, and ``take_profit_pips``.
    """

    def __init__(self, settings: Settings) -> None:
        self._fast = settings.sma_fast
        self._slow = settings.sma_slow
        self._sl_pips = settings.stop_loss_pips
        self._tp_pips = settings.take_profit_pips

    @property
    def name(self) -> str:
        """Return strategy name."""
        return 'sma_crossover'

    def required_bars(self) -> int:
        """Need at least slow-period + 2 bars for a valid crossover."""
        return self._slow + 2

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Analyse OHLCV data and emit a crossover signal.

        Args:
            df: OHLCV DataFrame with at least :meth:`required_bars` rows.
            symbol: Instrument symbol.

        Returns:
            A :class:`Signal` of type BUY, SELL, or HOLD.
        """
        if len(df) < self.required_bars():
            logger.warning(
                f'[SMA] Insufficient data for {symbol}: '
                f'{len(df)} bars < {self.required_bars()} required'
            )
            return Signal(SignalType.HOLD, symbol, 0.0, reason='Insufficient data')

        df = df.copy()
        df[f'sma_{self._fast}'] = ta.sma(df['close'], length=self._fast)
        df[f'sma_{self._slow}'] = ta.sma(df['close'], length=self._slow)
        df.dropna(inplace=True)

        if len(df) < 2:
            return Signal(
                SignalType.HOLD, symbol, 0.0, reason='Not enough data after SMA calculation'
            )

        fast_col = f'sma_{self._fast}'
        slow_col = f'sma_{self._slow}'

        prev_fast = df[fast_col].iloc[-2]
        prev_slow = df[slow_col].iloc[-2]
        curr_fast = df[fast_col].iloc[-1]
        curr_slow = df[slow_col].iloc[-1]
        price = float(df['close'].iloc[-1])

        # Calculate pip value (assume 4-decimal pairs; JPY pairs use 2)
        pip_size = 0.01 if 'JPY' in symbol else 0.0001

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            # Golden cross — bullish
            sl = price - self._sl_pips * pip_size
            tp = price + self._tp_pips * pip_size
            reason = (
                f'Golden cross: SMA{self._fast}={curr_fast:.5f} > SMA{self._slow}={curr_slow:.5f}'
            )
            logger.info(f'[SMA] BUY signal for {symbol} — {reason}')
            return Signal(
                SignalType.BUY,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=min(abs(curr_fast - curr_slow) / curr_slow * 100, 1.0),
                reason=reason,
                metadata={fast_col: curr_fast, slow_col: curr_slow},
            )

        if prev_fast >= prev_slow and curr_fast < curr_slow:
            # Death cross — bearish
            sl = price + self._sl_pips * pip_size
            tp = price - self._tp_pips * pip_size
            reason = (
                f'Death cross: SMA{self._fast}={curr_fast:.5f} < SMA{self._slow}={curr_slow:.5f}'
            )
            logger.info(f'[SMA] SELL signal for {symbol} — {reason}')
            return Signal(
                SignalType.SELL,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=min(abs(curr_fast - curr_slow) / curr_slow * 100, 1.0),
                reason=reason,
                metadata={fast_col: curr_fast, slow_col: curr_slow},
            )

        return Signal(SignalType.HOLD, symbol, price, reason='No crossover detected')
