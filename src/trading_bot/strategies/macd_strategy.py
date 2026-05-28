"""MACD histogram crossover strategy."""

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

    def plot_signal(self, df: pd.DataFrame, signal: Signal, symbol: str) -> 'go.Figure':
        """Two-panel chart: candlestick with SL/TP + MACD histogram, line, signal.

        Args:
            df: OHLCV DataFrame (full history; last 150 bars are displayed).
            signal: Signal to mark on the chart.
            symbol: Instrument name for the title.

        Returns:
            A :class:`plotly.graph_objects.Figure`.
        """
        import plotly.graph_objects as go
        import pandas_ta as ta
        from plotly.subplots import make_subplots

        display_n = min(len(df), 150)
        window = df.tail(display_n).copy()

        macd_result = ta.macd(df['close'], fast=self._fast, slow=self._slow, signal=self._signal)
        if macd_result is not None and not macd_result.empty:
            hist_col = next((c for c in macd_result.columns if c.startswith('MACDh')), None)
            line_col = next((c for c in macd_result.columns if c.startswith('MACD_')), None)
            sig_col  = next((c for c in macd_result.columns if c.startswith('MACDs')), None)
            if hist_col and line_col and sig_col:
                window['_hist'] = macd_result[hist_col].iloc[-display_n:].values
                window['_line'] = macd_result[line_col].iloc[-display_n:].values
                window['_sig']  = macd_result[sig_col].iloc[-display_n:].values

        has_macd = '_hist' in window.columns
        label = f'MACD({self._fast},{self._slow},{self._signal})'

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.6, 0.4],
            subplot_titles=[symbol, label],
        )

        # ── Row 1: price ──────────────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=window.index,
            open=window['open'], high=window['high'],
            low=window['low'], close=window['close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
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

        # ── Row 2: MACD ───────────────────────────────────────────────────
        if has_macd:
            hist_vals = window['_hist']
            colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist_vals]
            fig.add_trace(go.Bar(
                x=window.index, y=hist_vals, name='Histogram', marker_color=colors,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=window.index, y=window['_line'], name='MACD',
                line={'color': '#3498db', 'width': 1.5},
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=window.index, y=window['_sig'], name='Signal',
                line={'color': '#e67e22', 'width': 1.5},
            ), row=2, col=1)
            fig.add_hline(y=0, line_color='white', opacity=0.3, row=2, col=1)

        fig.update_layout(
            title=f'{symbol} — {label} | {signal.signal_type.value}',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=700,
            legend={'orientation': 'h', 'y': 1.02},
            margin={'l': 50, 'r': 80, 't': 60, 'b': 50},
        )
        return fig
