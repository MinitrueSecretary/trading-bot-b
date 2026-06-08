"""EMA-based trend direction analyser."""

from typing import TYPE_CHECKING, Literal

import pandas as pd

from trading_bot.analysers.base_analyser import Analysis, BaseAnalyser
from trading_bot.config import Settings
from trading_bot.utils.logger import get_logger

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = get_logger(__name__)

TrendDirection = Literal['uptrend', 'downtrend', 'sideways']


class EMATrendAnalyser(BaseAnalyser):
    """Classify the current trend using three exponential moving averages.

    Trend rules (applied to the most recent bar's EMA values):

    - **Uptrend**:   fast > medium > slow (fully aligned bullish)
    - **Downtrend**: fast < medium < slow (fully aligned bearish)
    - **Sideways**:  any other arrangement

    Args:
        settings: Application settings — reads ``ema_fast``, ``ema_medium``,
            and ``ema_slow``.
    """

    def __init__(self, settings: Settings) -> None:
        self._fast = settings.ema_fast
        self._medium = settings.ema_medium
        self._slow = settings.ema_slow

    @property
    def name(self) -> str:
        """Return analyser name."""
        return 'ema_trend'

    def required_bars(self) -> int:
        """Return minimum bars needed for stable EMA calculation."""
        return self._slow * 2

    def analyse(self, df: pd.DataFrame, symbol: str) -> Analysis:
        """Classify trend direction from the three EMAs.

        Args:
            df: OHLCV DataFrame (columns: ``open high low close volume``).
            symbol: Instrument symbol.

        Returns:
            :class:`Analysis` with ``trend``, ``ema_fast``, ``ema_medium``,
            and ``ema_slow`` values.
        """
        if len(df) < self.required_bars():
            logger.warning(
                f'[EMATrend] Insufficient data for {symbol}: '
                f'{len(df)} bars < {self.required_bars()} required'
            )
            return Analysis(
                symbol=symbol,
                data={'trend': 'unknown', 'ema_fast': None, 'ema_medium': None, 'ema_slow': None},
                notes='Insufficient data to calculate EMAs.',
            )

        close = df['close']
        ema_f = float(close.ewm(span=self._fast, adjust=False).mean().iloc[-1])
        ema_m = float(close.ewm(span=self._medium, adjust=False).mean().iloc[-1])
        ema_s = float(close.ewm(span=self._slow, adjust=False).mean().iloc[-1])

        if ema_f > ema_m > ema_s:
            trend: TrendDirection = 'uptrend'
        elif ema_f < ema_m < ema_s:
            trend = 'downtrend'
        else:
            trend = 'sideways'

        notes = (
            f'{trend.upper()} | '
            f'EMA{self._fast}={ema_f:.5f} '
            f'EMA{self._medium}={ema_m:.5f} '
            f'EMA{self._slow}={ema_s:.5f}'
        )
        logger.info(f'[EMATrend] {symbol} — {notes}')

        return Analysis(
            symbol=symbol,
            data={
                'trend': trend,
                f'ema_{self._fast}': round(ema_f, 5),
                f'ema_{self._medium}': round(ema_m, 5),
                f'ema_{self._slow}': round(ema_s, 5),
            },
            notes=notes,
            metadata={
                'ema_fast_period': self._fast,
                'ema_medium_period': self._medium,
                'ema_slow_period': self._slow,
            },
        )

    def plot(self, df: pd.DataFrame, analysis: Analysis, symbol: str) -> 'go.Figure':
        """Candlestick chart with three EMA lines overlaid.

        Args:
            df: Full OHLCV DataFrame (last 200 bars displayed).
            analysis: Analysis result from :meth:`analyse`.
            symbol: Instrument name for the chart title.

        Returns:
            A :class:`plotly.graph_objects.Figure`.
        """
        import plotly.graph_objects as go

        display_n = min(len(df), 200)
        window = df.tail(display_n).copy()

        close = df['close']
        window[f'_ema{self._fast}'] = (
            close.ewm(span=self._fast, adjust=False).mean().iloc[-display_n:].values
        )
        window[f'_ema{self._medium}'] = (
            close.ewm(span=self._medium, adjust=False).mean().iloc[-display_n:].values
        )
        window[f'_ema{self._slow}'] = (
            close.ewm(span=self._slow, adjust=False).mean().iloc[-display_n:].values
        )

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

        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window[f'_ema{self._fast}'],
                name=f'EMA{self._fast}',
                line={'color': '#3498db', 'width': 1.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window[f'_ema{self._medium}'],
                name=f'EMA{self._medium}',
                line={'color': '#e67e22', 'width': 1.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window[f'_ema{self._slow}'],
                name=f'EMA{self._slow}',
                line={'color': '#9b59b6', 'width': 1.5},
            )
        )

        fig.update_layout(
            title=f'{symbol} — EMA Trend {title_colour} {trend.upper()} | {analysis.notes}',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            legend={'orientation': 'h', 'y': 1.02},
            margin={'l': 50, 'r': 80, 't': 60, 'b': 50},
        )
        return fig
