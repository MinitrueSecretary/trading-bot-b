"""Swing-structure trend direction analyser."""

from typing import TYPE_CHECKING, Literal

import pandas as pd

from trading_bot.analysers.base_analyser import Analysis, BaseAnalyser
from trading_bot.config import Settings
from trading_bot.utils.logger import get_logger
from trading_bot.utils.pivots import swing_highs, swing_lows

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = get_logger(__name__)

TrendDirection = Literal['uptrend', 'downtrend', 'sideways']


class SwingTrendAnalyser(BaseAnalyser):
    """Classify the current trend by comparing successive swing highs and lows.

    Trend rules (evaluated on the last two confirmed swing highs and lows):

    - **Uptrend**:   higher high **and** higher low
    - **Downtrend**: lower high  **and** lower low
    - **Sideways**:  mixed or insufficient pivots

    At least two swing highs and two swing lows must be present.  If only one
    type has enough pivots, that signal is reported in ``notes`` but the
    direction falls back to ``sideways``.

    Args:
        settings: Application settings — reads ``rsi_divergence_lookback`` and
            ``rsi_divergence_swing_window``.
    """

    def __init__(self, settings: Settings) -> None:
        self._lookback = settings.rsi_divergence_lookback
        self._swing_window = settings.rsi_divergence_swing_window

    @property
    def name(self) -> str:
        """Return analyser name."""
        return 'swing_trend'

    def required_bars(self) -> int:
        """Return minimum bars needed to detect two pivot pairs."""
        return self._lookback + self._swing_window * 2

    def analyse(self, df: pd.DataFrame, symbol: str) -> Analysis:
        """Determine trend direction from swing-point structure.

        Args:
            df: OHLCV DataFrame (columns: ``open high low close volume``).
            symbol: Instrument symbol.

        Returns:
            :class:`Analysis` with ``trend``, ``swing_highs``, ``swing_lows``,
            ``hh_hl`` (higher-high / higher-low flags), and ``lh_ll`` flags.
        """
        if len(df) < self.required_bars():
            logger.warning(
                f'[SwingTrend] Insufficient data for {symbol}: '
                f'{len(df)} bars < {self.required_bars()} required'
            )
            return Analysis(
                symbol=symbol,
                data={'trend': 'unknown', 'swing_highs': [], 'swing_lows': []},
                notes='Insufficient data to detect swing pivots.',
            )

        search = df.iloc[-(self._lookback + self._swing_window) :]
        lows_arr = search['low'].to_numpy(dtype=float)
        highs_arr = search['high'].to_numpy(dtype=float)

        low_idx = swing_lows(lows_arr, self._swing_window)
        high_idx = swing_highs(highs_arr, self._swing_window)

        detected_lows = [round(float(lows_arr[i]), 5) for i in low_idx]
        detected_highs = [round(float(highs_arr[i]), 5) for i in high_idx]

        has_hh = has_hl = has_lh = has_ll = False
        high_signal = low_signal = ''

        if len(high_idx) >= 2:
            ph1, ph2 = float(highs_arr[high_idx[-2]]), float(highs_arr[high_idx[-1]])
            if ph2 > ph1:
                has_hh = True
                high_signal = f'HH ({ph1:.5f} → {ph2:.5f})'
            else:
                has_lh = True
                high_signal = f'LH ({ph1:.5f} → {ph2:.5f})'

        if len(low_idx) >= 2:
            pl1, pl2 = float(lows_arr[low_idx[-2]]), float(lows_arr[low_idx[-1]])
            if pl2 > pl1:
                has_hl = True
                low_signal = f'HL ({pl1:.5f} → {pl2:.5f})'
            else:
                has_ll = True
                low_signal = f'LL ({pl1:.5f} → {pl2:.5f})'

        if has_hh and has_hl:
            trend: TrendDirection = 'uptrend'
        elif has_lh and has_ll:
            trend = 'downtrend'
        else:
            trend = 'sideways'

        signals = ' | '.join(s for s in (high_signal, low_signal) if s)
        notes = (
            f'{trend.upper()} | {signals}' if signals else f'{trend.upper()} | insufficient pivots'
        )
        logger.info(f'[SwingTrend] {symbol} — {notes}')

        return Analysis(
            symbol=symbol,
            data={
                'trend': trend,
                'swing_highs': detected_highs,
                'swing_lows': detected_lows,
            },
            notes=notes,
            metadata={
                'has_hh': has_hh,
                'has_hl': has_hl,
                'has_lh': has_lh,
                'has_ll': has_ll,
                'swing_window': self._swing_window,
                'lookback': self._lookback,
            },
        )

    def plot(self, df: pd.DataFrame, analysis: Analysis, symbol: str) -> 'go.Figure':
        """Candlestick chart with swing pivot markers.

        Swing highs are marked with red downward triangles; swing lows with
        green upward triangles. The trend direction is shown in the title.

        Args:
            df: Full OHLCV DataFrame (last 200 bars displayed).
            analysis: Analysis result from :meth:`analyse`.
            symbol: Instrument name for the chart title.

        Returns:
            A :class:`plotly.graph_objects.Figure`.
        """
        import plotly.graph_objects as go

        display_n = min(len(df), 200)
        window = df.tail(display_n)

        search_n = self._lookback + self._swing_window
        search = window.iloc[-min(search_n, len(window)) :]
        lows_arr = search['low'].to_numpy(dtype=float)
        highs_arr = search['high'].to_numpy(dtype=float)
        low_idx = swing_lows(lows_arr, self._swing_window)
        high_idx = swing_highs(highs_arr, self._swing_window)

        trend = str(analysis.data.get('trend', 'sideways'))
        title_colour = {'uptrend': '🟢', 'downtrend': '🔴', 'sideways': '🟡'}.get(trend, '')

        fig = go.Figure()

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

        if low_idx:
            low_times = [search.index[i] for i in low_idx]
            low_vals = [lows_arr[i] for i in low_idx]
            fig.add_trace(
                go.Scatter(
                    x=low_times,
                    y=low_vals,
                    mode='markers',
                    marker={'symbol': 'triangle-up', 'size': 10, 'color': '#26a69a'},
                    name='Swing Low',
                )
            )

        if high_idx:
            high_times = [search.index[i] for i in high_idx]
            high_vals = [highs_arr[i] for i in high_idx]
            fig.add_trace(
                go.Scatter(
                    x=high_times,
                    y=high_vals,
                    mode='markers',
                    marker={'symbol': 'triangle-down', 'size': 10, 'color': '#ef5350'},
                    name='Swing High',
                )
            )

        # Connect last two swing lows and last two swing highs with a line
        # to make the structure direction visually obvious.
        if len(low_idx) >= 2:
            i1, i2 = low_idx[-2], low_idx[-1]
            fig.add_trace(
                go.Scatter(
                    x=[search.index[i1], search.index[i2]],
                    y=[lows_arr[i1], lows_arr[i2]],
                    mode='lines',
                    line={'color': '#26a69a', 'width': 2, 'dash': 'dot'},
                    showlegend=False,
                )
            )

        if len(high_idx) >= 2:
            i1, i2 = high_idx[-2], high_idx[-1]
            fig.add_trace(
                go.Scatter(
                    x=[search.index[i1], search.index[i2]],
                    y=[highs_arr[i1], highs_arr[i2]],
                    mode='lines',
                    line={'color': '#ef5350', 'width': 2, 'dash': 'dot'},
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f'{symbol} — Swing Trend {title_colour} {trend.upper()} | {analysis.notes}',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            legend={'orientation': 'h', 'y': 1.02},
            margin={'l': 50, 'r': 80, 't': 60, 'b': 50},
        )
        return fig
