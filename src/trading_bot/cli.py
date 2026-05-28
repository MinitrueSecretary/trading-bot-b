"""Command-line interface for the trading bot."""

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
@click.option('--strategy', default=None, help='Strategy name (overrides config).')
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
    help='Fixed lot size per trade (FixedLotSizeManager). Mutually exclusive with --risk-amount and --risk-pct.',
)
@click.option(
    '--risk-amount',
    'risk_amount',
    default=None,
    type=float,
    help='Fixed dollar amount at risk per trade (FixedRiskLotSizeManager). Mutually exclusive with --lot-size and --risk-pct.',
)
@click.option(
    '--risk-pct',
    'risk_pct',
    default=None,
    type=float,
    help='Percentage of balance to risk per trade (overrides BOT_RISK_PER_TRADE_PCT). Mutually exclusive with --lot-size and --risk-amount.',
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
    signal_discord: bool,
) -> None:
    """Start the live / paper trading loop."""
    from trading_bot.bot import TradingBot
    from trading_bot.utils.lot_size import (
        BaseLotSizeManager,
        FixedLotSizeManager,
        FixedRiskLotSizeManager,
    )

    exclusives = sum(x is not None for x in (lot_size, risk_amount, risk_pct))
    if exclusives > 1:
        raise click.UsageError('--lot-size, --risk-amount, and --risk-pct are mutually exclusive.')

    settings = get_settings()
    if dry_run is not None:
        settings = settings.model_copy(update={'dry_run': dry_run})
    if strategy is not None:
        settings = settings.model_copy(update={'strategy_name': strategy})
    if sl_pips is not None:
        settings = settings.model_copy(update={'stop_loss_pips': sl_pips})
    if tp_pips is not None:
        settings = settings.model_copy(update={'take_profit_pips': tp_pips})
    if risk_pct is not None:
        settings = settings.model_copy(update={'risk_per_trade_pct': risk_pct})

    lot_mgr: BaseLotSizeManager | None = None
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

    bot = TradingBot(settings, lot_size_manager=lot_mgr, signal_chart_discord=signal_discord)
    bot.start()


# ──────────────────────────────────────────────────────────────────────────────
# backtest command
# ──────────────────────────────────────────────────────────────────────────────


@main.command()
@click.option('--symbol', default=None, help='Symbol to backtest (e.g. EURUSD).')
@click.option('--strategy', default=None, help='Strategy name.')
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
    help='Fixed lot size per trade (FixedLotSizeManager). Mutually exclusive with --risk-amount.',
)
@click.option(
    '--risk-amount',
    'risk_amount',
    default=None,
    type=float,
    help='Fixed dollar amount at risk per trade (FixedRiskLotSizeManager). Mutually exclusive with --lot-size.',
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
    save_signal_chart: bool,
) -> None:
    """Run a strategy backtest and print statistics."""
    from trading_bot.backtesting import Backtester
    from trading_bot.data import get_data_provider
    from trading_bot.strategies import get_strategy
    from trading_bot.utils.lot_size import (
        BaseLotSizeManager,
        FixedLotSizeManager,
        FixedRiskLotSizeManager,
    )
    from trading_bot.visualization.charts import plot_backtest
    from trading_bot.visualization.charts import save_chart as save_chart_fn

    if lot_size is not None and risk_amount is not None:
        raise click.UsageError('--lot-size and --risk-amount are mutually exclusive.')

    settings = get_settings()
    if symbol:
        settings = settings.model_copy(update={'primary_symbol': symbol.upper()})
    if strategy:
        settings = settings.model_copy(update={'strategy_name': strategy})
    if sl_pips is not None:
        settings = settings.model_copy(update={'stop_loss_pips': sl_pips})
    if tp_pips is not None:
        settings = settings.model_copy(update={'take_profit_pips': tp_pips})

    # Resolve flag defaults from settings / env
    if save_chart is None:
        save_chart = settings.backtest_save_chart
    if save_csv is None:
        save_csv = settings.backtest_save_csv

    # Build lot size manager (None → Backtester uses its FixedLotSizeManager(0.1) default)
    lot_mgr: BaseLotSizeManager | None = None
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

    strat = get_strategy(settings)
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


if __name__ == '__main__':
    main()
