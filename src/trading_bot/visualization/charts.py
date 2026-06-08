"""Interactive Plotly charts for price analysis and backtest results."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pandas_ta as ta  # type: ignore[import]
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from trading_bot.backtesting.backtester import BacktestResult
    from trading_bot.strategies.base_strategy import Signal


def plot_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    signals: list[Signal] | None = None,
    sma_periods: list[int] | None = None,
    show_rsi: bool = False,
    show_volume: bool = True,
) -> go.Figure:
    """Build an interactive candlestick chart with optional overlays.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        symbol: Instrument name for the chart title.
        signals: Optional list of signals to plot as markers.
        sma_periods: List of SMA periods to overlay (e.g. ``[20, 50, 200]``).
        show_rsi: If ``True``, add an RSI panel below the candles.
        show_volume: If ``True``, add a volume panel.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    rows = 1
    row_heights = [0.7]
    subplot_titles = [symbol]

    if show_volume:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append('Volume')

    if show_rsi:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append('RSI(14)')

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Candlestick ─────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1,
        col=1,
    )

    # ── SMA overlays ─────────────────────────────────────────────────────
    sma_colors = ['#f39c12', '#3498db', '#9b59b6', '#1abc9c']
    for idx, period in enumerate(sma_periods or []):
        sma_vals = ta.sma(df['close'], length=period)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_vals,
                name=f'SMA{period}',
                line={'color': sma_colors[idx % len(sma_colors)], 'width': 1.5},
            ),
            row=1,
            col=1,
        )

    # ── Signal markers ────────────────────────────────────────────────────
    if signals:
        buy_signals = [s for s in signals if s.signal_type.value == 'BUY']
        sell_signals = [s for s in signals if s.signal_type.value == 'SELL']

        if buy_signals:
            # Find the closest bar index for each signal entry price
            fig.add_trace(
                go.Scatter(
                    x=[s.metadata.get('bar_time', df.index[-1]) for s in buy_signals],
                    y=[s.entry_price for s in buy_signals],
                    mode='markers',
                    marker={'symbol': 'triangle-up', 'size': 12, 'color': '#26a69a'},
                    name='BUY',
                ),
                row=1,
                col=1,
            )
        if sell_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.metadata.get('bar_time', df.index[-1]) for s in sell_signals],
                    y=[s.entry_price for s in sell_signals],
                    mode='markers',
                    marker={'symbol': 'triangle-down', 'size': 12, 'color': '#ef5350'},
                    name='SELL',
                ),
                row=1,
                col=1,
            )

    current_row = 2

    # ── Volume ────────────────────────────────────────────────────────────
    if show_volume:
        colors = [
            '#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'], strict=True)
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
            row=current_row,
            col=1,
        )
        current_row += 1

    # ── RSI ───────────────────────────────────────────────────────────────
    if show_rsi:
        rsi_vals = ta.rsi(df['close'], length=14)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=rsi_vals, name='RSI(14)', line={'color': '#e67e22', 'width': 1.5}
            ),
            row=current_row,
            col=1,
        )
        # Overbought / oversold lines
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=current_row, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=current_row, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color='gray', opacity=0.4, row=current_row, col=1)

    fig.update_layout(
        title=f'{symbol} — Price Chart',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=700,
        legend={'orientation': 'h', 'y': 1.02},
        margin={'l': 50, 'r': 50, 't': 60, 'b': 50},
    )

    return fig


def plot_backtest(result: BacktestResult) -> go.Figure:
    """Build a multi-panel backtest results chart.

    Panels:
    1. Equity curve vs buy-and-hold.
    2. Per-trade P&L bar chart.

    Args:
        result: A completed :class:`~trading_bot.backtesting.backtester.BacktestResult`.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=['Equity Curve', 'Trade P&L'],
        row_heights=[0.6, 0.4],
    )

    # ── Equity curve ──────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name='Strategy Equity',
            line={'color': '#3498db', 'width': 2},
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)',
        ),
        row=1,
        col=1,
    )

    # Baseline: flat at initial balance
    fig.add_hline(
        y=result.initial_balance,
        line_dash='dash',
        line_color='gray',
        opacity=0.6,
        row=1,
        col=1,
    )

    # ── Per-trade P&L ─────────────────────────────────────────────────────
    if not result.trades.empty and 'pnl' in result.trades.columns:
        pnl_vals = result.trades['pnl'].astype(float)
        colors = ['#26a69a' if v > 0 else '#ef5350' for v in pnl_vals]
        labels = (
            result.trades['entry_index'].astype(str)
            if 'entry_index' in result.trades.columns
            else list(range(len(pnl_vals)))
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=pnl_vals,
                name='Trade P&L',
                marker_color=colors,
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_color='white', opacity=0.3, row=2, col=1)

    fig.update_layout(
        title=f'Backtest: {result.strategy_name} | {result.symbol} | Return: {result.total_return_pct:+.1f}%',
        template='plotly_dark',
        height=700,
        showlegend=True,
        margin={'l': 50, 'r': 50, 't': 60, 'b': 50},
    )

    return fig


def show(fig: go.Figure) -> None:
    """Open the figure in the default browser.

    Args:
        fig: Plotly figure to display.
    """
    fig.show()


def save_png(
    fig: go.Figure,
    symbol: str,
    label: str = 'signal',
    output_dir: str | Path = 'signals',
    width: int = 1200,
    height: int = 700,
) -> Path:
    """Save *fig* as a PNG image and return the path.

    Uses `kaleido <https://github.com/plotly/Kaleido>`_ for static export.
    The filename follows the same ``{symbol}_{label}_{UTC-timestamp}.png``
    convention as :func:`save_chart`.

    Args:
        fig: Plotly figure to export.
        symbol: Instrument name used in the filename.
        label: Short tag describing the chart type (e.g. ``'signal'``).
        output_dir: Folder to write into (default: ``signals/`` relative to cwd).
        width: Image width in pixels (default: 1200).
        height: Image height in pixels (default: 700).

    Returns:
        :class:`~pathlib.Path` of the saved PNG.
    """
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')
    safe_symbol = re.sub(r'[^\w]', '_', symbol)
    dest = folder / f'{safe_symbol}_{label}_{ts}.png'

    # Kaleido's orjson serializer does not support pd.Timestamp; round-trip
    # through plotly's own JSON serializer (which does) to normalise all
    # timestamp values to ISO strings before handing the figure to kaleido.
    import plotly.io as pio

    fig = pio.from_json(fig.to_json())

    fig.write_image(str(dest), width=width, height=height)
    return dest


def save_html(fig: go.Figure, path: str) -> None:
    """Save the figure as a standalone HTML file.

    Args:
        fig: Plotly figure to save.
        path: Output file path (e.g. ``'chart.html'``).
    """
    fig.write_html(path)


def save_chart(
    fig: go.Figure,
    symbol: str,
    label: str = 'chart',
    output_dir: str | Path = 'charts',
) -> Path:
    """Save *fig* as an HTML file inside *output_dir* and return the path.

    The filename is built as ``{symbol}_{label}_{UTC-timestamp}.html`` so
    successive runs never overwrite each other.  The directory is created
    automatically if it does not exist.

    Args:
        fig: Plotly figure to save.
        symbol: Instrument name used in the filename (e.g. ``'EURUSD'``).
        label: Short tag describing the chart type (e.g. ``'analyse'``).
        output_dir: Folder to write into (default: ``charts/`` relative to cwd).

    Returns:
        :class:`~pathlib.Path` of the saved file.
    """
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')
    safe_symbol = re.sub(r'[^\w]', '_', symbol)
    filename = f'{safe_symbol}_{label}_{ts}.html'
    dest = folder / filename

    fig.write_html(str(dest))
    return dest
