"""MACD histogram crossover strategy."""

import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.utils.logger import get_logger
from trading_bot.utils.tpsl import BaseTPSLManager, FixedPipsTPSLManager

logger = get_logger(__name__)


class MACDStrategy(BaseStrategy):
    """MACD histogram zero-line crossover strategy.

    Entry rules:
    - **BUY**:  MACD histogram crosses from negative to positive.
    - **SELL**: MACD histogram crosses from positive to negative.

    Args:
        settings: Application settings providing ``macd_fast``, ``macd_slow``,
            ``macd_signal``, ``stop_loss_pips``, ``take_profit_pips``.
    """

    def __init__(
        self,
        settings: Settings,
        tpsl_manager: BaseTPSLManager | None = None,
    ) -> None:
        self._fast = settings.macd_fast
        self._slow = settings.macd_slow
        self._signal = settings.macd_signal
        self._tpsl = tpsl_manager or FixedPipsTPSLManager(
            sl_pips=settings.stop_loss_pips,
            tp_pips=settings.take_profit_pips,
        )

    @property
    def name(self) -> str:
        """Return strategy name."""
        return 'macd'

    def required_bars(self) -> int:
        """Minimum bars for MACD calculation."""
        return self._slow + self._signal + 2

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate a signal based on MACD histogram zero-line crossover.

        Args:
            df: OHLCV DataFrame.
            symbol: Instrument symbol.

        Returns:
            A :class:`Signal`.
        """
        if len(df) < self.required_bars():
            return Signal(SignalType.HOLD, symbol, 0.0, reason='Insufficient data')

        df = df.copy()
        macd_result = ta.macd(
            df['close'],
            fast=self._fast,
            slow=self._slow,
            signal=self._signal,
        )

        if macd_result is None or macd_result.empty:
            return Signal(SignalType.HOLD, symbol, 0.0, reason='MACD calculation failed')

        # pandas-ta returns columns like MACDh_12_26_9
        hist_col = [c for c in macd_result.columns if c.startswith('MACDh')]
        macd_col = [c for c in macd_result.columns if c.startswith('MACD_')]
        if not hist_col or not macd_col:
            return Signal(SignalType.HOLD, symbol, 0.0, reason='Unexpected MACD column names')

        df['macd_hist'] = macd_result[hist_col[0]]
        df['macd_line'] = macd_result[macd_col[0]]
        df.dropna(inplace=True)

        if len(df) < 2:
            return Signal(SignalType.HOLD, symbol, 0.0, reason='Insufficient post-calculation data')

        price = float(df['close'].iloc[-1])
        hist_prev = float(df['macd_hist'].iloc[-2])
        hist_curr = float(df['macd_hist'].iloc[-1])
        macd_curr = float(df['macd_line'].iloc[-1])

        if hist_prev < 0 and hist_curr >= 0:
            sl = self._tpsl.calculate_sl(SignalType.BUY, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.BUY, price, symbol)
            reason = (
                f'MACD histogram zero-cross bullish: {hist_prev:.6f} → {hist_curr:.6f} '
                f'| MACD={macd_curr:.6f}'
            )
            logger.info(f'[MACD] BUY signal for {symbol} — {reason}')
            return Signal(
                SignalType.BUY,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=min(abs(hist_curr) * 1000, 1.0),
                reason=reason,
                metadata={'macd_hist': hist_curr, 'macd_line': macd_curr},
            )

        if hist_prev > 0 and hist_curr <= 0:
            sl = self._tpsl.calculate_sl(SignalType.SELL, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.SELL, price, symbol)
            reason = (
                f'MACD histogram zero-cross bearish: {hist_prev:.6f} → {hist_curr:.6f} '
                f'| MACD={macd_curr:.6f}'
            )
            logger.info(f'[MACD] SELL signal for {symbol} — {reason}')
            return Signal(
                SignalType.SELL,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=min(abs(hist_curr) * 1000, 1.0),
                reason=reason,
                metadata={'macd_hist': hist_curr, 'macd_line': macd_curr},
            )

        return Signal(
            SignalType.HOLD,
            symbol,
            price,
            reason=f'MACD hist={hist_curr:.6f} (no zero-cross)',
        )
