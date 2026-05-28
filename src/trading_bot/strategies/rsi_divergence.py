"""RSI Divergence strategy using swing-point pivot detection."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from trading_bot.config import Settings

if TYPE_CHECKING:
    import plotly.graph_objects as go
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.utils.logger import get_logger
from trading_bot.utils.tpsl import BaseTPSLManager, FixedPipsTPSLManager

logger = get_logger(__name__)


def _swing_lows(series: npt.NDArray[np.float64], window: int) -> list[int]:
    """Return indices of confirmed swing lows in *series*.

    A bar at index *i* is a swing low when its value is strictly less than
    every bar within ``window`` positions to the left **and** all available
    bars to the right (up to ``window``).  Bars near the right edge use
    however many bars exist rather than requiring a full ``window``, so
    pivots can be detected all the way up to the final bar.

    Args:
        series: 1-D array of values to scan.
        window: Number of bars required to the left; right side uses
            ``min(window, available)`` bars.

    Returns:
        Sorted list of integer indices that are confirmed swing lows.
    """
    indices: list[int] = []
    n = len(series)
    # Stop at n-2: the last bar (n-1) is excluded because it may be an
    # unfinished candle and cannot yet be confirmed as a local minimum.
    for i in range(window, n - 1):
        val = series[i]
        right = min(window, n - 1 - i)
        if all(val < series[i - j] for j in range(1, window + 1)) and all(
            val < series[i + j] for j in range(1, right + 1)
        ):
            indices.append(i)
    return indices


def _swing_highs(series: npt.NDArray[np.float64], window: int) -> list[int]:
    """Return indices of confirmed swing highs in *series*.

    A bar at index *i* is a swing high when its value is strictly greater
    than every bar within ``window`` positions to the left **and** all
    available bars to the right (up to ``window``).  The last bar (*n-1*)
    is excluded as it may be an unfinished candle.

    Args:
        series: 1-D array of values to scan.
        window: Number of bars required to the left; right side uses
            ``min(window, available)`` bars.

    Returns:
        Sorted list of integer indices that are confirmed swing highs.
    """
    indices: list[int] = []
    n = len(series)
    # Stop at n-2 for the same reason as _swing_lows.
    for i in range(window, n - 1):
        val = series[i]
        right = min(window, n - 1 - i)
        if all(val > series[i - j] for j in range(1, window + 1)) and all(
            val > series[i + j] for j in range(1, right + 1)
        ):
            indices.append(i)
    return indices


class RSIDivergenceStrategy(BaseStrategy):
    """RSI Divergence strategy — detects divergence via swing-point pivots.

    Finds the last two confirmed swing lows / highs within a lookback window
    for both price and the Stochastic RSI K-line, then classifies one of four
    divergence types:

    - **Regular Bullish** (BUY):  price ``LL`` AND RSI ``HL``
    - **Hidden  Bullish** (BUY):  price ``HL`` AND RSI ``LL``
    - **Regular Bearish** (SELL): price ``HH`` AND RSI ``LH``
    - **Hidden  Bearish** (SELL): price ``LH`` AND RSI ``HH``

    Priority order: regular bullish -> regular bearish -> hidden bullish ->
    hidden bearish.  First match wins.

    Args:
        settings: Application settings.
        tpsl_manager: Optional SL/TP manager (defaults to
            :class:`~trading_bot.utils.tpsl.FixedPipsTPSLManager`).
    """

    def __init__(
        self,
        settings: Settings,
        tpsl_manager: BaseTPSLManager | None = None,
    ) -> None:
        self._rsi_period = settings.stoch_rsi_period
        self._k = settings.stoch_rsi_k
        self._d = settings.stoch_rsi_d
        self._lookback = settings.rsi_divergence_lookback
        self._swing_window = settings.rsi_divergence_swing_window
        self._tpsl = tpsl_manager or FixedPipsTPSLManager(
            sl_pips=settings.stop_loss_pips,
            tp_pips=settings.take_profit_pips,
        )

    @property
    def name(self) -> str:
        """Return strategy name."""
        return 'rsi_divergence'

    def required_bars(self) -> int:
        """Return minimum bars needed for a valid signal.

        StochRSI needs ~``2 * rsi_period`` bars to stabilise.  The pivot
        search adds ``lookback`` bars, plus ``swing_window`` bars of
        confirmation buffer on either edge.
        """
        return 2 * self._rsi_period + self._lookback + self._swing_window

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Detect RSI divergence via swing pivots and return a :class:`Signal`.

        Args:
            df: OHLCV DataFrame (columns: ``open high low close volume``).
            symbol: Instrument symbol.

        Returns:
            A :class:`Signal` — BUY, SELL, or HOLD.
        """
        if len(df) < self.required_bars():
            logger.warning(
                f'[RSIDivergence] Insufficient data for {symbol}: '
                f'{len(df)} bars < {self.required_bars()} required'
            )
            return Signal(SignalType.HOLD, symbol, 0.0, reason='Insufficient data')

        df = df.copy()

        # ── Compute Stochastic RSI ───────────────────────────────────────────
        stochrsi = ta.stochrsi(
            df['close'],
            length=self._rsi_period,
            rsi_length=self._rsi_period,
            k=self._k,
            d=self._d,
        )
        if stochrsi is None or stochrsi.empty:
            return Signal(
                SignalType.HOLD, symbol, 0.0, reason='StochRSI calculation returned no data'
            )

        k_col = stochrsi.columns[0]  # STOCHRSIk_…
        d_col = stochrsi.columns[1]  # STOCHRSId_…
        df['stochrsi_k'] = stochrsi[k_col].values
        df['stochrsi_d'] = stochrsi[d_col].values
        df.dropna(subset=['stochrsi_k', 'stochrsi_d'], inplace=True)

        if len(df) < self._lookback + self._swing_window:
            return Signal(
                SignalType.HOLD,
                symbol,
                0.0,
                reason='Not enough data after StochRSI calculation',
            )

        # ── Slice the pivot search window ────────────────────────────────────
        # Keep a window large enough for swing confirmation on both edges.
        search_slice = df.iloc[-(self._lookback + self._swing_window) :]

        price_lows = search_slice['low'].to_numpy(dtype=float)
        price_highs = search_slice['high'].to_numpy(dtype=float)
        rsi_k = search_slice['stochrsi_k'].to_numpy(dtype=float)

        price = float(df['close'].iloc[-1])
        stochrsi_k_curr = float(df['stochrsi_k'].iloc[-1])
        stochrsi_d_curr = float(df['stochrsi_d'].iloc[-1])

        # ── Find confirmed pivots ────────────────────────────────────────────
        low_idx = _swing_lows(price_lows, self._swing_window)
        high_idx = _swing_highs(price_highs, self._swing_window)

        # Freshness filter: P2 must be the latest candles.
        # Bar n-1 is excluded from pivot detection (unfinished candle), so the
        # newest possible pivot is n-2.
        # If 3 latest candles -> index ≥ n-4.
        freshness_min = len(price_lows) - 2
        low_fresh = len(low_idx) >= 2 and low_idx[-1] >= freshness_min
        high_fresh = len(high_idx) >= 2 and high_idx[-1] >= freshness_min

        base_metadata: dict[str, object] = {
            'stochrsi_k': round(stochrsi_k_curr, 4),
            'stochrsi_d': round(stochrsi_d_curr, 4),
        }

        # ── Lows-based divergence checks (need ≥ 2 fresh swing lows) ─────────
        if low_fresh:
            i1, i2 = low_idx[-2], low_idx[-1]  # i1 older, i2 newer
            p1_low, p2_low = float(price_lows[i1]), float(price_lows[i2])
            r1_low, r2_low = float(rsi_k[i1]), float(rsi_k[i2])

            # Regular Bullish: price LL, RSI HL
            if p2_low < p1_low and r2_low > r1_low:
                sl = self._tpsl.calculate_sl(SignalType.BUY, price, symbol)
                tp = self._tpsl.calculate_tp(SignalType.BUY, price, symbol)
                reason = (
                    f'Regular bullish divergence: '
                    f'price LL ({p1_low:.5f} -> {p2_low:.5f}) | '
                    f'RSI HL ({r1_low:.1f} -> {r2_low:.1f})'
                )
                logger.info(f'[RSIDivergence] BUY signal for {symbol} — {reason}')
                return Signal(
                    SignalType.BUY,
                    symbol,
                    price,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=min(abs(p1_low - p2_low) / p1_low + abs(r2_low - r1_low) / 100.0, 1.0),
                    reason=reason,
                    metadata={
                        **base_metadata,
                        'divergence_type': 'regular_bullish',
                        'price_pivot1': round(p1_low, 5),
                        'price_pivot2': round(p2_low, 5),
                        'rsi_pivot1': round(r1_low, 4),
                        'rsi_pivot2': round(r2_low, 4),
                    },
                )

        # ── Highs-based divergence checks (need ≥ 2 fresh swing highs) ─────
        if high_fresh:
            i1, i2 = high_idx[-2], high_idx[-1]
            p1_high, p2_high = float(price_highs[i1]), float(price_highs[i2])
            r1_high, r2_high = float(rsi_k[i1]), float(rsi_k[i2])

            # Regular Bearish: price HH, RSI LH
            if p2_high > p1_high and r2_high < r1_high:
                sl = self._tpsl.calculate_sl(SignalType.SELL, price, symbol)
                tp = self._tpsl.calculate_tp(SignalType.SELL, price, symbol)
                reason = (
                    f'Regular bearish divergence: '
                    f'price HH ({p1_high:.5f} -> {p2_high:.5f}) | '
                    f'RSI LH ({r1_high:.1f} -> {r2_high:.1f})'
                )
                logger.info(f'[RSIDivergence] SELL signal for {symbol} — {reason}')
                return Signal(
                    SignalType.SELL,
                    symbol,
                    price,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=min(
                        abs(p2_high - p1_high) / p1_high + abs(r1_high - r2_high) / 100.0, 1.0
                    ),
                    reason=reason,
                    metadata={
                        **base_metadata,
                        'divergence_type': 'regular_bearish',
                        'price_pivot1': round(p1_high, 5),
                        'price_pivot2': round(p2_high, 5),
                        'rsi_pivot1': round(r1_high, 4),
                        'rsi_pivot2': round(r2_high, 4),
                    },
                )

        # ── Hidden divergences (lower priority) ─────────────────────────────
        if low_fresh:
            i1, i2 = low_idx[-2], low_idx[-1]
            p1_low, p2_low = float(price_lows[i1]), float(price_lows[i2])
            r1_low, r2_low = float(rsi_k[i1]), float(rsi_k[i2])

            # Hidden Bullish: price HL, RSI LL
            if p2_low > p1_low and r2_low < r1_low:
                sl = self._tpsl.calculate_sl(SignalType.BUY, price, symbol)
                tp = self._tpsl.calculate_tp(SignalType.BUY, price, symbol)
                reason = (
                    f'Hidden bullish divergence: '
                    f'price HL ({p1_low:.5f} -> {p2_low:.5f}) | '
                    f'RSI LL ({r1_low:.1f} -> {r2_low:.1f})'
                )
                logger.info(f'[RSIDivergence] BUY signal for {symbol} — {reason}')
                return Signal(
                    SignalType.BUY,
                    symbol,
                    price,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=min(abs(p2_low - p1_low) / p1_low + abs(r1_low - r2_low) / 100.0, 1.0),
                    reason=reason,
                    metadata={
                        **base_metadata,
                        'divergence_type': 'hidden_bullish',
                        'price_pivot1': round(p1_low, 5),
                        'price_pivot2': round(p2_low, 5),
                        'rsi_pivot1': round(r1_low, 4),
                        'rsi_pivot2': round(r2_low, 4),
                    },
                )

        if high_fresh:
            i1, i2 = high_idx[-2], high_idx[-1]
            p1_high, p2_high = float(price_highs[i1]), float(price_highs[i2])
            r1_high, r2_high = float(rsi_k[i1]), float(rsi_k[i2])

            # Hidden Bearish: price LH, RSI HH
            if p2_high < p1_high and r2_high > r1_high:
                sl = self._tpsl.calculate_sl(SignalType.SELL, price, symbol)
                tp = self._tpsl.calculate_tp(SignalType.SELL, price, symbol)
                reason = (
                    f'Hidden bearish divergence: '
                    f'price LH ({p1_high:.5f} -> {p2_high:.5f}) | '
                    f'RSI HH ({r1_high:.1f} -> {r2_high:.1f})'
                )
                logger.info(f'[RSIDivergence] SELL signal for {symbol} — {reason}')
                return Signal(
                    SignalType.SELL,
                    symbol,
                    price,
                    stop_loss=sl,
                    take_profit=tp,
                    strength=min(
                        abs(p1_high - p2_high) / p1_high + abs(r2_high - r1_high) / 100.0, 1.0
                    ),
                    reason=reason,
                    metadata={
                        **base_metadata,
                        'divergence_type': 'hidden_bearish',
                        'price_pivot1': round(p1_high, 5),
                        'price_pivot2': round(p2_high, 5),
                        'rsi_pivot1': round(r1_high, 4),
                        'rsi_pivot2': round(r2_high, 4),
                    },
                )

        stale_note = ''
        if (len(low_idx) >= 2 and not low_fresh) or (len(high_idx) >= 2 and not high_fresh):
            stale_note = ' (pivots found but P2 not within last 3 candles)'
        return Signal(
            SignalType.HOLD,
            symbol,
            price,
            reason=(
                f'No divergence detected{stale_note} | '
                f'swing_lows={len(low_idx)} swing_highs={len(high_idx)} '
                f'StochRSI-K={stochrsi_k_curr:.1f}'
            ),
        )

    def plot_signal(self, df: pd.DataFrame, signal: Signal, symbol: str) -> 'go.Figure':
        """Two-panel chart: candlestick with divergence tangents (top) and StochRSI K/D (bottom).

        When the signal carries pivot metadata, trend lines are drawn on both
        panels connecting the two divergence pivot bars — making the
        price/RSI disagreement visually obvious.

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

        stochrsi = ta.stochrsi(
            df['close'],
            length=self._rsi_period,
            rsi_length=self._rsi_period,
            k=self._k,
            d=self._d,
        )
        if stochrsi is not None and not stochrsi.empty:
            k_col, d_col = stochrsi.columns[0], stochrsi.columns[1]
            window['_k'] = stochrsi[k_col].iloc[-display_n:].values
            window['_d'] = stochrsi[d_col].iloc[-display_n:].values

        has_stochrsi = '_k' in window.columns
        k_label = f'StochRSI-K({self._rsi_period},{self._k},{self._d})'

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.6, 0.4],
            subplot_titles=[symbol, k_label],
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

            # ── Divergence tangent lines ───────────────────────────────────
            # Re-detect pivots over the search slice so we can get bar timestamps.
            if has_stochrsi and len(window) >= self._lookback + self._swing_window:
                search = window.iloc[-(self._lookback + self._swing_window):]
                is_low_div = 'bullish' in str(signal.metadata.get('divergence_type', ''))
                rsi_k_arr = search['_k'].to_numpy(dtype=float)

                if is_low_div:
                    price_arr = search['low'].to_numpy(dtype=float)
                    pivot_fn = _swing_lows
                else:
                    price_arr = search['high'].to_numpy(dtype=float)
                    pivot_fn = _swing_highs

                pivots = pivot_fn(price_arr, self._swing_window)
                if len(pivots) >= 2:
                    i1, i2 = pivots[-2], pivots[-1]
                    t1, t2 = search.index[i1], search.index[i2]
                    p1_val = price_arr[i1]
                    p2_val = price_arr[i2]
                    r1_val = rsi_k_arr[i1]
                    r2_val = rsi_k_arr[i2]
                    tang_colour = '#26a69a' if is_buy else '#ef5350'

                    # Price tangent
                    fig.add_trace(go.Scatter(
                        x=[t1, t2], y=[p1_val, p2_val],
                        mode='lines+markers',
                        line={'color': tang_colour, 'width': 2, 'dash': 'dot'},
                        marker={'size': 7, 'color': tang_colour},
                        name='Price trend',
                        showlegend=False,
                    ), row=1, col=1)

                    # RSI tangent (drawn in row 2 below after RSI traces added)
                    fig.add_trace(go.Scatter(
                        x=[t1, t2], y=[r1_val, r2_val],
                        mode='lines+markers',
                        line={'color': tang_colour, 'width': 2, 'dash': 'dot'},
                        marker={'size': 7, 'color': tang_colour},
                        name='RSI trend',
                        showlegend=False,
                    ), row=2, col=1)

        # ── Row 2: StochRSI ───────────────────────────────────────────────
        if has_stochrsi:
            fig.add_trace(go.Scatter(
                x=window.index, y=window['_k'],
                name='K', line={'color': '#3498db', 'width': 1.5},
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=window.index, y=window['_d'],
                name='D', line={'color': '#e67e22', 'width': 1.5},
            ), row=2, col=1)
            fig.add_hline(y=80, line_dash='dash', line_color='#ef5350',
                          annotation_text='80', annotation_position='right', row=2, col=1)
            fig.add_hline(y=20, line_dash='dash', line_color='#26a69a',
                          annotation_text='20', annotation_position='right', row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=1)

        div_type = signal.metadata.get('divergence_type', 'none') if signal.is_actionable else 'HOLD'
        fig.update_layout(
            title=f'{symbol} — RSI Divergence | {div_type} | {signal.signal_type.value}',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=700,
            legend={'orientation': 'h', 'y': 1.02},
            margin={'l': 50, 'r': 80, 't': 60, 'b': 50},
        )
        return fig
