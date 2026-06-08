"""Support and resistance analyser using swing-point pivot detection."""

from typing import TYPE_CHECKING

import pandas as pd

from trading_bot.analysers.base_analyser import Analysis, BaseAnalyser
from trading_bot.config import Settings
from trading_bot.utils.logger import get_logger
from trading_bot.utils.pivots import swing_highs, swing_lows

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = get_logger(__name__)

_MAX_LEVELS = 5


class SupportResistanceAnalyser(BaseAnalyser):
    """Identify support and resistance levels from swing-point pivots.

    Scans the most recent ``lookback`` bars (plus ``swing_window`` confirmation
    buffer) for confirmed swing lows (support) and swing highs (resistance),
    then returns up to :data:`_MAX_LEVELS` of each, sorted ascending.

    Args:
        settings: Application settings — reads ``rsi_divergence_lookback`` and
            ``rsi_divergence_swing_window`` so no new config keys are required.
    """

    def __init__(self, settings: Settings) -> None:
        self._lookback = settings.rsi_divergence_lookback
        self._swing_window = settings.rsi_divergence_swing_window

    @property
    def name(self) -> str:
        """Return analyser name."""
        return 'support_resistance'

    def required_bars(self) -> int:
        """Return minimum bars needed to detect at least one pivot pair."""
        return self._lookback + self._swing_window * 2

    def analyse(self, df: pd.DataFrame, symbol: str) -> Analysis:
        """Detect support and resistance levels via swing pivots.

        Args:
            df: OHLCV DataFrame (columns: ``open high low close volume``).
            symbol: Instrument symbol.

        Returns:
            :class:`Analysis` with ``support`` and ``resistance`` level lists.
        """
        if len(df) < self.required_bars():
            logger.warning(
                f'[SupportResistance] Insufficient data for {symbol}: '
                f'{len(df)} bars < {self.required_bars()} required'
            )
            return Analysis(
                symbol=symbol,
                data={'support': [], 'resistance': []},
                notes='Insufficient data to detect pivots.',
            )

        search = df.iloc[-(self._lookback + self._swing_window) :]
        lows_arr = search['low'].to_numpy(dtype=float)
        highs_arr = search['high'].to_numpy(dtype=float)

        low_idx = swing_lows(lows_arr, self._swing_window)
        high_idx = swing_highs(highs_arr, self._swing_window)

        support_levels = sorted({round(float(lows_arr[i]), 5) for i in low_idx[-_MAX_LEVELS:]})
        resistance_levels = sorted({round(float(highs_arr[i]), 5) for i in high_idx[-_MAX_LEVELS:]})

        current_price = float(df['close'].iloc[-1])

        nearest_sup = max((s for s in support_levels if s < current_price), default=None)
        nearest_res = min((r for r in resistance_levels if r > current_price), default=None)

        notes_parts: list[str] = [f'Price @ {current_price:.5f}']
        if nearest_sup is not None:
            notes_parts.append(f'nearest support {nearest_sup:.5f}')
        if nearest_res is not None:
            notes_parts.append(f'nearest resistance {nearest_res:.5f}')

        logger.info(
            f'[SupportResistance] {symbol} — '
            f'{len(support_levels)} support, {len(resistance_levels)} resistance levels detected'
        )

        return Analysis(
            symbol=symbol,
            data={
                'support': support_levels,
                'resistance': resistance_levels,
            },
            notes=' | '.join(notes_parts),
            metadata={
                'swing_window': self._swing_window,
                'lookback': self._lookback,
                'bars_used': len(search),
            },
        )

    def plot(self, df: pd.DataFrame, analysis: Analysis, symbol: str) -> 'go.Figure':
        """Candlestick chart with dashed S/R lines overlaid.

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

        support_raw = analysis.data.get('support', [])
        resistance_raw = analysis.data.get('resistance', [])
        support_levels = [float(v) for v in (support_raw if isinstance(support_raw, list) else [])]
        resistance_levels = [
            float(v) for v in (resistance_raw if isinstance(resistance_raw, list) else [])
        ]

        for level in support_levels:
            fig.add_hline(
                y=level,
                line_dash='dash',
                line_color='#26a69a',
                annotation_text=f'S {level:.5f}',
                annotation_position='right',
            )

        for level in resistance_levels:
            fig.add_hline(
                y=level,
                line_dash='dash',
                line_color='#ef5350',
                annotation_text=f'R {level:.5f}',
                annotation_position='right',
            )

        fig.update_layout(
            title=f'{symbol} — Support & Resistance | {analysis.notes}',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            margin={'l': 50, 'r': 120, 't': 60, 'b': 50},
        )
        return fig
