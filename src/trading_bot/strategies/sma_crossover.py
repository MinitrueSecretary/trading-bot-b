"""SMA Crossover strategy — fast MA crosses above/below slow MA."""

from typing import TYPE_CHECKING

import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from trading_bot.config import Settings

if TYPE_CHECKING:
    import plotly.graph_objects as go
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.utils.logger import get_logger
from trading_bot.utils.tpsl import BaseTPSLManager, FixedPipsTPSLManager

logger = get_logger(__name__)


class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy.

    Generates a **BUY** signal when the fast SMA crosses above the slow SMA,
    and a **SELL** signal on the opposite crossover.

    Args:
        settings: Application settings providing ``sma_fast``, ``sma_slow``,
            ``stop_loss_pips``, and ``take_profit_pips``.
    """

    def __init__(
        self,
        settings: Settings,
        tpsl_manager: BaseTPSLManager | None = None,
    ) -> None:
        self._fast = settings.sma_fast
        self._slow = settings.sma_slow
        self._tpsl = tpsl_manager or FixedPipsTPSLManager(
            sl_pips=settings.stop_loss_pips,
            tp_pips=settings.take_profit_pips,
        )

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

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            # Golden cross — bullish
            sl = self._tpsl.calculate_sl(SignalType.BUY, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.BUY, price, symbol)
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
            sl = self._tpsl.calculate_sl(SignalType.SELL, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.SELL, price, symbol)
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

    def plot_signal(self, df: pd.DataFrame, signal: Signal, symbol: str) -> 'go.Figure':
        """Candlestick chart with fast/slow SMA overlays and signal marker.

        Args:
            df: OHLCV DataFrame (full history; last 100 bars are displayed).
            signal: Signal to mark on the chart.
            symbol: Instrument name for the title.

        Returns:
            A :class:`plotly.graph_objects.Figure`.
        """
        import pandas_ta as ta
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        display_n = min(len(df), 100)
        fast_col = f'sma_{self._fast}'
        slow_col = f'sma_{self._slow}'

        # Compute on full df for accuracy; slice to display window
        fast_sma = ta.sma(df['close'], length=self._fast)
        slow_sma = ta.sma(df['close'], length=self._slow)
        window = df.tail(display_n).copy()
        window[fast_col] = fast_sma.iloc[-display_n:].values
        window[slow_col] = slow_sma.iloc[-display_n:].values

        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(
            go.Candlestick(
                x=window.index,
                open=window['open'],
                high=window['high'],
                low=window['low'],
                close=window['close'],
                name=symbol,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window[fast_col],
                name=f'SMA{self._fast}',
                line={'color': '#f39c12', 'width': 1.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window[slow_col],
                name=f'SMA{self._slow}',
                line={'color': '#3498db', 'width': 1.5},
            )
        )

        if signal.is_actionable:
            is_buy = signal.signal_type == SignalType.BUY
            fig.add_trace(
                go.Scatter(
                    x=[window.index[-1]],
                    y=[signal.entry_price],
                    mode='markers',
                    marker={
                        'symbol': 'triangle-up' if is_buy else 'triangle-down',
                        'size': 14,
                        'color': '#26a69a' if is_buy else '#ef5350',
                    },
                    name=signal.signal_type.value,
                )
            )
            if signal.stop_loss:
                fig.add_hline(
                    y=signal.stop_loss,
                    line_dash='dash',
                    line_color='#ef5350',
                    annotation_text='SL',
                    annotation_position='right',
                )
            if signal.take_profit:
                fig.add_hline(
                    y=signal.take_profit,
                    line_dash='dash',
                    line_color='#26a69a',
                    annotation_text='TP',
                    annotation_position='right',
                )

        fig.update_layout(
            title=f'{symbol} — SMA {self._fast}/{self._slow} | {signal.signal_type.value}',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            legend={'orientation': 'h', 'y': 1.02},
            margin={'l': 50, 'r': 80, 't': 60, 'b': 50},
        )
        return fig
