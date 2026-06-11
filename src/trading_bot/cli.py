"""Command-line interface for the trading bot."""

from datetime import datetime

import click
from rich.table import Table

from trading_bot.config import get_settings
from trading_bot.utils.logger import console, get_logger

logger = get_logger(__name__)


@click.group()
@click.version_option(version='0.1.0')
def main() -> None:
    """Forex Trading Bot — MT5 / YFinance cross-platform."""


# ──────────────────────────────────────────────────────────────────────────────
# trade command
# ──────────────────────────────────────────────────────────────────────────────


@main.command()
@click.option('--dry-run/--live', default=None, help='Override dry_run setting.')
@click.option('--strategy', default=None, help='Strategy name. Mutually exclusive with --config.')
@click.option(
    '--sl-pips',
    'sl_pips',
    default=None,
    type=float,
    help='Stop-loss distance in pips (overrides BOT_STOP_LOSS_PIPS).',
)
@click.option(
    '--tp-pips',
    'tp_pips',
    default=None,
    type=float,
    help='Take-profit distance in pips (overrides BOT_TAKE_PROFIT_PIPS).',
)
@click.option(
    '--lot-size',
    'lot_size',
    default=None,
    type=float,
    help='Fixed lot size per trade. Mutually exclusive with --risk-amount, --risk-pct, and --config.',
)
@click.option(
    '--risk-amount',
    'risk_amount',
    default=None,
    type=float,
    help='Fixed dollar amount at risk per trade. Mutually exclusive with --lot-size, --risk-pct, and --config.',
)
@click.option(
    '--risk-pct',
    'risk_pct',
    default=None,
    type=float,
    help='Percentage of balance to risk per trade (overrides BOT_RISK_PER_TRADE_PCT). Mutually exclusive with --lot-size, --risk-amount, and --config.',
)
@click.option(
    '--config',
    'bot_config',
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        'Path to a Python bot config file (e.g. bot_config.py). '
        'May define build_strategy(settings) and/or build_lot_size_manager(). '
        'Mutually exclusive with --strategy, --lot-size, --risk-amount, and --risk-pct.'
    ),
)
@click.option(
    '--signal-discord/--no-signal-discord',
    'signal_discord',
    default=False,
    help='Send a strategy-specific signal chart to Discord on each actionable signal (requires BOT_DISCORD_WEBHOOK_URL and kaleido).',
)
def trade(
    dry_run: bool | None,
    strategy: str | None,
    sl_pips: float | None,
    tp_pips: float | None,
    lot_size: float | None,
    risk_amount: float | None,
    risk_pct: float | None,
    bot_config: str | None,
    signal_discord: bool,
) -> None:
    """Start the live / paper trading loop."""
    from trading_bot.bot import TradingBot
    from trading_bot.utils.bot_config_loader import load_bot_config
    from trading_bot.utils.lot_size import (
        BaseLotSizeManager,
        FixedLotSizeManager,
        FixedRiskLotSizeManager,
    )

    if bot_config is not None:
        conflicts = [
            n
            for n, v in (
                ('--strategy', strategy),
                ('--lot-size', lot_size),
                ('--risk-amount', risk_amount),
                ('--risk-pct', risk_pct),
            )
            if v is not None
        ]
        if conflicts:
            raise click.UsageError(f'--config is mutually exclusive with: {", ".join(conflicts)}')
    else:
        exclusives = sum(x is not None for x in (lot_size, risk_amount, risk_pct))
        if exclusives > 1:
            raise click.UsageError(
                '--lot-size, --risk-amount, and --risk-pct are mutually exclusive.'
            )

    settings = get_settings()
    if dry_run is not None:
        settings = settings.model_copy(update={'dry_run': dry_run})
    if sl_pips is not None:
        settings = settings.model_copy(update={'stop_loss_pips': sl_pips})
    if tp_pips is not None:
        settings = settings.model_copy(update={'take_profit_pips': tp_pips})

    composite_strat = None
    lot_mgr: BaseLotSizeManager | None = None

    if bot_config is not None:
        composite_strat, lot_mgr = load_bot_config(bot_config, settings)
        if composite_strat is not None:
            console.print(
                f'[cyan]Strategy: CompositeStrategy ({len(composite_strat.entries)} entries, '
                f'mode={composite_strat.mode}) (from {bot_config})[/]'
            )
        if lot_mgr is not None:
            console.print(f'[cyan]Lot sizing: {type(lot_mgr).__name__} (from {bot_config})[/]')
        else:
            pct = settings.risk_per_trade_pct
            console.print(f'[cyan]Lot sizing: percentage risk ({pct}% of balance)[/]')
    else:
        if strategy is not None:
            settings = settings.model_copy(update={'strategy_name': strategy})
        if risk_pct is not None:
            settings = settings.model_copy(update={'risk_per_trade_pct': risk_pct})
        if lot_size is not None:
            lot_mgr = FixedLotSizeManager(lot_size=lot_size)
            console.print(f'[cyan]Lot sizing: FixedLotSizeManager (lot_size={lot_size})[/]')
        elif risk_amount is not None:
            lot_mgr = FixedRiskLotSizeManager(risk_amount=risk_amount)
            console.print(
                f'[cyan]Lot sizing: FixedRiskLotSizeManager (risk_amount=${risk_amount:,.2f})[/]'
            )
        else:
            pct = settings.risk_per_trade_pct
            console.print(f'[cyan]Lot sizing: percentage risk ({pct}% of balance)[/]')

    bot = TradingBot(
        settings,
        lot_size_manager=lot_mgr,
        signal_chart_discord=signal_discord,
        strategy=composite_strat,
    )
    bot.start()


# ──────────────────────────────────────────────────────────────────────────────
# backtest command
# ──────────────────────────────────────────────────────────────────────────────


@main.command()
@click.option('--symbol', default=None, help='Symbol to backtest (e.g. EURUSD).')
@click.option('--strategy', default=None, help='Strategy name. Mutually exclusive with --config.')
@click.option('--bars', default=1000, show_default=True, help='Number of bars to fetch.')
@click.option(
    '--save-chart/--no-save-chart',
    'save_chart',
    default=None,
    help='Build and save the HTML chart (default: BOT_BACKTEST_SAVE_CHART).',
)
@click.option(
    '--save-csv/--no-save-csv',
    'save_csv',
    default=None,
    help='Save trades DataFrame to CSV (default: BOT_BACKTEST_SAVE_CSV).',
)
@click.option(
    '--sl-pips',
    'sl_pips',
    default=None,
    type=float,
    help='Stop-loss distance in pips for FixedPipsTPSLManager (overrides BOT_STOP_LOSS_PIPS).',
)
@click.option(
    '--tp-pips',
    'tp_pips',
    default=None,
    type=float,
    help='Take-profit distance in pips for FixedPipsTPSLManager (overrides BOT_TAKE_PROFIT_PIPS).',
)
@click.option(
    '--lot-size',
    'lot_size',
    default=None,
    type=float,
    help='Fixed lot size per trade. Mutually exclusive with --risk-amount and --config.',
)
@click.option(
    '--risk-amount',
    'risk_amount',
    default=None,
    type=float,
    help='Fixed dollar amount at risk per trade. Mutually exclusive with --lot-size and --config.',
)
@click.option(
    '--config',
    'bot_config',
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        'Path to a Python bot config file (e.g. bot_config.py). '
        'May define build_strategy(settings) and/or build_lot_size_manager(). '
        'Mutually exclusive with --strategy, --lot-size, and --risk-amount.'
    ),
)
@click.option(
    '--save-signal-chart/--no-save-signal-chart',
    'save_signal_chart',
    default=False,
    help='Save a strategy-specific signal PNG after the backtest (requires kaleido).',
)
def backtest(
    symbol: str | None,
    strategy: str | None,
    bars: int,
    save_chart: bool | None,
    save_csv: bool | None,
    sl_pips: float | None,
    tp_pips: float | None,
    lot_size: float | None,
    risk_amount: float | None,
    bot_config: str | None,
    save_signal_chart: bool,
) -> None:
    """Run a strategy backtest and print statistics."""
    from trading_bot.backtesting import Backtester
    from trading_bot.data import get_data_provider
    from trading_bot.strategies import get_strategy
    from trading_bot.utils.bot_config_loader import load_bot_config
    from trading_bot.utils.lot_size import (
        BaseLotSizeManager,
        FixedLotSizeManager,
        FixedRiskLotSizeManager,
    )
    from trading_bot.visualization.charts import plot_backtest
    from trading_bot.visualization.charts import save_chart as save_chart_fn

    if bot_config is not None:
        conflicts = [
            n
            for n, v in (
                ('--strategy', strategy),
                ('--lot-size', lot_size),
                ('--risk-amount', risk_amount),
            )
            if v is not None
        ]
        if conflicts:
            raise click.UsageError(f'--config is mutually exclusive with: {", ".join(conflicts)}')
    else:
        if lot_size is not None and risk_amount is not None:
            raise click.UsageError('--lot-size and --risk-amount are mutually exclusive.')

    settings = get_settings()
    if symbol:
        settings = settings.model_copy(update={'primary_symbol': symbol.upper()})
    if sl_pips is not None:
        settings = settings.model_copy(update={'stop_loss_pips': sl_pips})
    if tp_pips is not None:
        settings = settings.model_copy(update={'take_profit_pips': tp_pips})

    # Resolve flag defaults from settings / env
    if save_chart is None:
        save_chart = settings.backtest_save_chart
    if save_csv is None:
        save_csv = settings.backtest_save_csv

    composite_strat = None
    lot_mgr: BaseLotSizeManager | None = None

    if bot_config is not None:
        composite_strat, lot_mgr = load_bot_config(bot_config, settings)
        if composite_strat is not None:
            console.print(
                f'[cyan]Strategy: CompositeStrategy ({len(composite_strat.entries)} entries, '
                f'mode={composite_strat.mode}) (from {bot_config})[/]'
            )
        if lot_mgr is not None:
            console.print(f'[cyan]Lot sizing: {type(lot_mgr).__name__} (from {bot_config})[/]')
    else:
        if strategy:
            settings = settings.model_copy(update={'strategy_name': strategy})
        if risk_amount is not None:
            lot_mgr = FixedRiskLotSizeManager(risk_amount=risk_amount)
            console.print(
                f'[cyan]Lot sizing: FixedRiskLotSizeManager (risk_amount=${risk_amount:,.2f})[/]'
            )
        elif lot_size is not None:
            lot_mgr = FixedLotSizeManager(lot_size=lot_size)
            console.print(f'[cyan]Lot sizing: FixedLotSizeManager (lot_size={lot_size})[/]')

    provider = get_data_provider(settings)
    provider.connect()

    yf_symbol = settings.yf_symbol if not settings.is_windows else settings.primary_symbol

    console.print(
        f'[bold cyan]Fetching {bars} bars for {settings.primary_symbol} '
        f'(tf={settings.timeframe.value})...[/]'
    )
    df = provider.get_ohlcv(yf_symbol, settings.timeframe, bars=bars)
    provider.disconnect()

    console.print(f'[green]Got {len(df)} bars ({df.index[0]} → {df.index[-1]})[/]')

    strat = composite_strat if composite_strat is not None else get_strategy(settings)
    bt = Backtester(strat, lot_size_manager=lot_mgr)
    result = bt.run(df, settings.primary_symbol)

    console.print(result.summary())

    if save_csv:
        dest = result.save_trades_csv()
        console.print(f'[green]Trades saved → {dest}[/]')

    if save_chart:
        fig = plot_backtest(result)
        dest = save_chart_fn(fig, settings.primary_symbol, label='backtest')
        console.print(f'[green]Chart saved → {dest}[/]')

    if save_signal_chart:
        from trading_bot.visualization.charts import save_png

        console.print('[cyan]Generating signal chart...[/]')
        signal = strat.generate_signal(df, settings.primary_symbol)
        fig = strat.plot_signal(df, signal, settings.primary_symbol)
        dest = save_png(fig, settings.primary_symbol, label='signal')
        console.print(f'[green]Signal chart saved → {dest}[/]')


# ──────────────────────────────────────────────────────────────────────────────
# analyse command
# ──────────────────────────────────────────────────────────────────────────────


@main.command()
@click.option('--symbol', default=None, help='Symbol to analyse.')
@click.option('--bars', default=300, show_default=True, help='Number of bars.')
@click.option('--chart/--no-chart', default=True, show_default=True, help='Open chart.')
@click.option(
    '--sl-pips',
    'sl_pips',
    default=None,
    type=float,
    help='Stop-loss distance in pips (overrides BOT_STOP_LOSS_PIPS).',
)
@click.option(
    '--tp-pips',
    'tp_pips',
    default=None,
    type=float,
    help='Take-profit distance in pips (overrides BOT_TAKE_PROFIT_PIPS).',
)
def analyse(
    symbol: str | None, bars: int, chart: bool, sl_pips: float | None, tp_pips: float | None
) -> None:
    """Fetch live data, generate a signal, and optionally show a chart."""
    from trading_bot.data import get_data_provider
    from trading_bot.strategies import get_strategy
    from trading_bot.visualization.charts import plot_ohlcv, save_chart

    settings = get_settings()
    if symbol:
        settings = settings.model_copy(update={'primary_symbol': symbol.upper()})
    if sl_pips is not None:
        settings = settings.model_copy(update={'stop_loss_pips': sl_pips})
    if tp_pips is not None:
        settings = settings.model_copy(update={'take_profit_pips': tp_pips})

    provider = get_data_provider(settings)
    provider.connect()

    yf_symbol = settings.yf_symbol if not settings.is_windows else settings.primary_symbol
    console.print(f'[cyan]Fetching {bars} bars for {settings.primary_symbol}...[/]')

    df = provider.get_ohlcv(yf_symbol, settings.timeframe, bars=bars)

    strat = get_strategy(settings)
    signal = strat.generate_signal(df, settings.primary_symbol)

    # Pretty-print signal
    colour = (
        'green'
        if signal.signal_type.value == 'BUY'
        else 'red'
        if signal.signal_type.value == 'SELL'
        else 'yellow'
    )
    console.print(
        f'\n[bold {colour}]{signal.signal_type.value}[/bold {colour}] signal for '
        f'[bold]{signal.symbol}[/bold]'
    )
    console.print(f'  Entry     : {signal.entry_price:.5f}')
    console.print(f'  Stop Loss : {signal.stop_loss:.5f}')
    console.print(f'  Take Profit: {signal.take_profit:.5f}')
    console.print(f'  Reason    : {signal.reason}')

    provider.disconnect()

    if chart:
        sma_periods = [settings.sma_fast, settings.sma_slow]
        fig = plot_ohlcv(
            df,
            settings.primary_symbol,
            sma_periods=sma_periods,
            show_rsi=True,
            show_volume=True,
        )
        dest = save_chart(fig, settings.primary_symbol, label='analyse')
        console.print(f'[green]Chart saved → {dest}[/]')


# ──────────────────────────────────────────────────────────────────────────────
# fetch-ohlcv command
# ──────────────────────────────────────────────────────────────────────────────


@main.command('fetch-ohlcv')
@click.option('--symbol', default=None, help='Symbol in MT5 format (default: BOT_PRIMARY_SYMBOL).')
@click.option(
    '--timeframe',
    default=None,
    type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk']),
    help='OHLCV timeframe (default: BOT_TIMEFRAME).',
)
@click.option(
    '--bars',
    default=500,
    show_default=True,
    help='Number of bars to fetch (ignored when --start is used).',
)
@click.option(
    '--start',
    'start_dt',
    default=None,
    type=click.DateTime(formats=['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']),
    help='Range start (UTC). E.g. 2024-01-01 or 2024-01-01T08:00:00.',
)
@click.option(
    '--end',
    'end_dt',
    default=None,
    type=click.DateTime(formats=['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']),
    help='Range end (UTC, default: now). Requires --start.',
)
@click.option(
    '--output',
    'output_path',
    default=None,
    type=click.Path(),
    help='Destination CSV file. Auto-generates ohlcv/{SYMBOL}_{TIMESTAMP}.csv if omitted.',
)
def fetch_ohlcv(
    symbol: str | None,
    timeframe: str | None,
    bars: int,
    start_dt: datetime | None,
    end_dt: datetime | None,
    output_path: str | None,
) -> None:
    """Fetch OHLCV data from MT5 (or yfinance on non-Windows) and save as CSV.

    Use --bars for the most recent N candles, or --start / --end for a
    specific date range (UTC).  --start and --end take precedence over --bars.
    """
    import re
    from datetime import UTC
    from pathlib import Path

    from trading_bot.config import Timeframe
    from trading_bot.data import get_data_provider

    # click.DateTime returns naive datetime objects — treat as UTC
    start: datetime | None = (
        start_dt.replace(tzinfo=UTC) if start_dt is not None else None
    )
    end: datetime | None = None
    if end_dt is not None:
        if start is None:
            raise click.UsageError('--end requires --start.')
        end = end_dt.replace(tzinfo=UTC)

    settings = get_settings()
    sym = symbol.upper() if symbol else settings.primary_symbol
    tf = Timeframe(timeframe) if timeframe else settings.timeframe

    provider = get_data_provider(settings)
    provider.connect()

    fetch_sym = settings.yf_symbol_for(sym) if not settings.is_windows else sym

    if start is not None:
        end_display = end.date() if end else 'now'
        console.print(f'[cyan]Fetching {sym} {tf.value} from {start.date()} to {end_display}...[/]')
    else:
        console.print(f'[cyan]Fetching {bars} bars of {sym} {tf.value}...[/]')

    try:
        df = provider.get_ohlcv(fetch_sym, tf, bars=bars, start=start, end=end)
    finally:
        provider.disconnect()

    if output_path:
        dest = Path(output_path)
    else:
        ts = datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')
        safe_sym = re.sub(r'[^\w]', '_', sym)
        dest = Path('ohlcv') / f'{safe_sym}_{tf.value.replace("/", "")}_{ts}.csv'

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest)

    console.print(f'[green]OK[/green] {len(df)} bars saved -> [bold]{dest}[/bold]')


# ──────────────────────────────────────────────────────────────────────────────
# ping-discord command
# ──────────────────────────────────────────────────────────────────────────────


@main.command('ping-discord')
def ping_discord() -> None:
    """Send a test ping to the configured Discord webhook and report the result."""
    from trading_bot.notifications.discord import DiscordNotifier

    settings = get_settings()

    if not settings.discord_webhook_url:
        console.print('[bold red]✗[/bold red] No Discord webhook URL configured.')
        console.print('  Set [cyan]BOT_DISCORD_WEBHOOK_URL[/cyan] in your [cyan].env[/cyan] file.')
        raise SystemExit(1)

    console.print('[cyan]Pinging Discord webhook…[/cyan]')
    notifier = DiscordNotifier(settings.discord_webhook_url)
    ok = notifier.ping()

    if ok:
        console.print('[bold green]✓[/bold green] Ping delivered — check your Discord channel.')
    else:
        console.print('[bold red]✗[/bold red] Ping failed — verify the webhook URL and network.')
        raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────────────────
# strategies command
# ──────────────────────────────────────────────────────────────────────────────


@main.command('list-strategies')
def list_strategies_cmd() -> None:
    """List all available strategies."""
    from trading_bot.strategies import list_strategies

    table = Table(title='Available Strategies', show_header=True)
    table.add_column('Name', style='cyan')
    table.add_column('Description')

    descriptions = {
        'sma_crossover': 'Simple Moving Average golden/death cross',
        'rsi': 'RSI overbought/oversold with 200-SMA trend filter',
        'macd': 'MACD histogram zero-line crossover',
    }

    for name in list_strategies():
        table.add_row(name, descriptions.get(name, '—'))

    console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# study command
# ──────────────────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    '--analyser',
    'analyser_name',
    default='support_resistance',
    show_default=True,
    help='Analyser to run.',
)
@click.option('--symbol', default=None, help='Symbol to study (default: BOT_PRIMARY_SYMBOL).')
@click.option('--bars', default=300, show_default=True, help='Number of bars to fetch.')
@click.option(
    '--timeframe',
    default=None,
    type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk']),
    help='OHLCV timeframe (default: BOT_TIMEFRAME).',
)
@click.option('--chart/--no-chart', default=False, show_default=True, help='Save HTML chart.')
def study(
    analyser_name: str, symbol: str | None, bars: int, timeframe: str | None, chart: bool
) -> None:
    """Fetch live data, run an analyser, and print the findings."""
    from trading_bot.analysers import get_analyser
    from trading_bot.config import Timeframe
    from trading_bot.data import get_data_provider
    from trading_bot.visualization.charts import save_chart

    settings = get_settings()
    if symbol:
        settings = settings.model_copy(update={'primary_symbol': symbol.upper()})
    tf = Timeframe(timeframe) if timeframe else settings.timeframe

    provider = get_data_provider(settings)
    provider.connect()

    yf_symbol = settings.yf_symbol if not settings.is_windows else settings.primary_symbol
    console.print(f'[cyan]Fetching {bars} bars ({tf.value}) for {settings.primary_symbol}...[/]')

    df = provider.get_ohlcv(yf_symbol, tf, bars=bars)
    provider.disconnect()

    analyser = get_analyser(analyser_name, settings)
    analysis = analyser.analyse(df, settings.primary_symbol)

    console.print(f'\n[bold cyan]{analyser_name}[/bold cyan] — [bold]{analysis.symbol}[/bold]')
    for key, val in analysis.data.items():
        console.print(f'  {key}: {val}')
    if analysis.notes:
        console.print(f'  [dim]{analysis.notes}[/dim]')

    if chart:
        fig = analyser.plot(df, analysis, settings.primary_symbol)
        dest = save_chart(fig, settings.primary_symbol, label=analyser_name)
        console.print(f'[green]Chart saved → {dest}[/]')


# ──────────────────────────────────────────────────────────────────────────────
# list-analysers command
# ──────────────────────────────────────────────────────────────────────────────


@main.command('list-analysers')
def list_analysers_cmd() -> None:
    """List all available analysers."""
    from trading_bot.analysers import list_analysers

    table = Table(title='Available Analysers', show_header=True)
    table.add_column('Name', style='cyan')
    table.add_column('Description')

    descriptions = {
        'support_resistance': 'Swing-pivot support and resistance levels',
        'ema_trend': 'Three-EMA alignment — uptrend / downtrend / sideways',
        'swing_trend': 'Higher-high/higher-low swing structure trend classification',
    }

    for name in list_analysers():
        table.add_row(name, descriptions.get(name, '—'))

    console.print(table)


if __name__ == '__main__':
    main()
