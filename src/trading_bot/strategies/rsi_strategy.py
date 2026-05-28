"""RSI overbought/oversold strategy with trend filter."""

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

    def plot_signal(self, df: pd.DataFrame, signal: Signal, symbol: str) -> 'go.Figure':
        """Two-panel chart: candlestick + SMA200 (top) and RSI with OB/OS bands (bottom).

        Divergence is highlighted visually through the overbought/oversold threshold lines
        and the signal marker, making the trigger condition immediately obvious.

        Args:
            df: OHLCV DataFrame (full history; last 200 bars are displayed).
            signal: Signal to mark on the chart.
            symbol: Instrument name for the title.

        Returns:
            A :class:`plotly.graph_objects.Figure`.
        """
        import plotly.graph_objects as go
        import pandas_ta as ta
        from plotly.subplots import make_subplots

        display_n = min(len(df), 200)
        window = df.tail(display_n).copy()

        # Compute indicators on full df; slice to display window
        rsi_vals = ta.rsi(df['close'], length=self._period)
        sma200_vals = ta.sma(df['close'], length=self._TREND_PERIOD)
        window['_rsi'] = rsi_vals.iloc[-display_n:].values
        window['_sma200'] = sma200_vals.iloc[-display_n:].values

        rsi_label = f'RSI({self._period})'

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.6, 0.4],
            subplot_titles=[symbol, rsi_label],
        )

        # ── Row 1: price + SMA200 ─────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=window.index,
            open=window['open'], high=window['high'],
            low=window['low'], close=window['close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=window.index, y=window['_sma200'],
            name='SMA200', line={'color': '#9b59b6', 'width': 1.5},
        ), row=1, col=1)

        if signal.is_actionable:
            is_buy = signal.signal_type == SignalType.BUY
            fig.add_trace(go.Scatter(
                x=[window.index[-1]], y=[signal.entry_price],
                mode='markers',
                marker={
                    'symbol': 'triangle-up' if is_buy else 'triangle-down',
                    'size': 14,
                    'color': '#26a69a' if is_buy else '#ef5350',
                },
                name=signal.signal_type.value,
            ), row=1, col=1)
            if signal.stop_loss:
                fig.add_hline(y=signal.stop_loss, line_dash='dash', line_color='#ef5350',
                              annotation_text='SL', annotation_position='right', row=1, col=1)
            if signal.take_profit:
                fig.add_hline(y=signal.take_profit, line_dash='dash', line_color='#26a69a',
                              annotation_text='TP', annotation_position='right', row=1, col=1)

        # ── Row 2: RSI ────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=window.index, y=window['_rsi'],
            name=rsi_label, line={'color': '#e67e22', 'width': 1.5},
        ), row=2, col=1)

        # Filled region between RSI and the triggered threshold for visual clarity
        if signal.is_actionable:
            is_buy = signal.signal_type == SignalType.BUY
            threshold = self._oversold if is_buy else self._overbought
            rsi_series = window['_rsi'].fillna(50)
            fill_y = rsi_series.clip(upper=threshold) if is_buy else rsi_series.clip(lower=threshold)
            fig.add_trace(go.Scatter(
                x=window.index, y=fill_y,
                fill='tonexty', fillcolor='rgba(230, 126, 34, 0.15)',
                line={'width': 0}, showlegend=False,
            ), row=2, col=1)

        fig.add_hline(y=self._overbought, line_dash='dash', line_color='#ef5350',
                      annotation_text=str(int(self._overbought)), annotation_position='right',
                      row=2, col=1)
        fig.add_hline(y=self._oversold, line_dash='dash', line_color='#26a69a',
                      annotation_text=str(int(self._oversold)), annotation_position='right',
                      row=2, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color='gray', opacity=0.4, row=2, col=1)

        fig.update_yaxes(range=[0, 100], row=2, col=1)

        fig.update_layout(
            title=f'{symbol} — {rsi_label} | {signal.signal_type.value}',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=700,
            legend={'orientation': 'h', 'y': 1.02},
            margin={'l': 50, 'r': 80, 't': 60, 'b': 50},
        )
        return fig
