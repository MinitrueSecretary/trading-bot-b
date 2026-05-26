"""RSI overbought/oversold strategy with trend filter."""

import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.utils.logger import get_logger
from trading_bot.utils.tpsl import BaseTPSLManager, FixedPipsTPSLManager

logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    """RSI Overbought / Oversold strategy with 200-SMA trend filter.

    Entry rules:
    - **BUY**:  RSI crosses above ``rsi_oversold`` AND price > SMA200 (uptrend).
    - **SELL**: RSI crosses below ``rsi_overbought`` AND price < SMA200 (downtrend).

    Args:
        settings: Application settings.
    """

    _TREND_PERIOD = 200

    def __init__(
        self,
        settings: Settings,
        tpsl_manager: BaseTPSLManager | None = None,
    ) -> None:
        self._period = settings.rsi_period
        self._overbought = settings.rsi_overbought
        self._oversold = settings.rsi_oversold
        self._tpsl = tpsl_manager or FixedPipsTPSLManager(
            sl_pips=settings.stop_loss_pips,
            tp_pips=settings.take_profit_pips,
        )

    @property
    def name(self) -> str:
        """Return strategy name."""
        return 'rsi'

    def required_bars(self) -> int:
        """Require enough bars for the trend SMA."""
        return self._TREND_PERIOD + self._period + 2

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate a signal based on RSI extremes and trend direction.

        Args:
            df: OHLCV DataFrame.
            symbol: Instrument symbol.

        Returns:
            A :class:`Signal`.
        """
        if len(df) < self.required_bars():
            logger.warning(
                f'[RSI] Insufficient data for {symbol}: '
                f'{len(df)} bars < {self.required_bars()} required'
            )
            return Signal(SignalType.HOLD, symbol, 0.0, reason='Insufficient data')

        df = df.copy()
        df['rsi'] = ta.rsi(df['close'], length=self._period)
        df['sma200'] = ta.sma(df['close'], length=self._TREND_PERIOD)
        df.dropna(inplace=True)

        if len(df) < 2:
            return Signal(
                SignalType.HOLD, symbol, 0.0, reason='Not enough data after indicator calculation'
            )

        price = float(df['close'].iloc[-1])
        rsi_prev = float(df['rsi'].iloc[-2])
        rsi_curr = float(df['rsi'].iloc[-1])
        sma200 = float(df['sma200'].iloc[-1])

        uptrend = price > sma200

        if rsi_prev <= self._oversold and rsi_curr > self._oversold and uptrend:
            sl = self._tpsl.calculate_sl(SignalType.BUY, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.BUY, price, symbol)
            reason = (
                f'RSI crossed above oversold ({self._oversold}): '
                f'{rsi_prev:.1f} -> {rsi_curr:.1f} | price > SMA200={sma200:.5f}'
            )
            logger.info(f'[RSI] BUY signal for {symbol} — {reason}')
            return Signal(
                SignalType.BUY,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=(self._oversold - rsi_prev) / self._oversold,
                reason=reason,
                metadata={'rsi': rsi_curr, 'sma200': sma200},
            )

        if rsi_prev >= self._overbought and rsi_curr < self._overbought and not uptrend:
            sl = self._tpsl.calculate_sl(SignalType.SELL, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.SELL, price, symbol)
            reason = (
                f'RSI crossed below overbought ({self._overbought}): '
                f'{rsi_prev:.1f} -> {rsi_curr:.1f} | price < SMA200={sma200:.5f}'
            )
            logger.info(f'[RSI] SELL signal for {symbol} — {reason}')
            return Signal(
                SignalType.SELL,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=(rsi_prev - self._overbought) / (100 - self._overbought),
                reason=reason,
                metadata={'rsi': rsi_curr, 'sma200': sma200},
            )

        return Signal(
            SignalType.HOLD,
            symbol,
            price,
            reason=f'RSI={rsi_curr:.1f} | trend={"up" if uptrend else "down"}',
        )
