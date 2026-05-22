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
def trade(dry_run: bool | None, strategy: str | None) -> None:
    """Start the live / paper trading loop."""
    from trading_bot.bot import TradingBot

    settings = get_settings()
    if dry_run is not None:
        settings = settings.model_copy(update={'dry_run': dry_run})
    if strategy is not None:
        settings = settings.model_copy(update={'strategy_name': strategy})

    bot = TradingBot(settings)
    bot.start()


# ──────────────────────────────────────────────────────────────────────────────
# backtest command
# ──────────────────────────────────────────────────────────────────────────────


@main.command()
@click.option('--symbol', default=None, help='Symbol to backtest (e.g. EURUSD).')
@click.option('--strategy', default=None, help='Strategy name.')
@click.option('--bars', default=1000, show_default=True, help='Number of bars to fetch.')
@click.option('--chart/--no-chart', default=True, show_default=True, help='Open results chart.')
@click.option('--save-html', 'save_html', default=None, help='Save chart to HTML file.')
def backtest(
    symbol: str | None,
    strategy: str | None,
    bars: int,
    chart: bool,
    save_html: str | None,
) -> None:
    """Run a strategy backtest and print statistics."""
    from trading_bot.backtesting import Backtester
    from trading_bot.data import get_data_provider
    from trading_bot.strategies import get_strategy
    from trading_bot.visualization.charts import plot_backtest, show
    from trading_bot.visualization.charts import save_html as save_html_fn

    settings = get_settings()
    if symbol:
        settings = settings.model_copy(update={'primary_symbol': symbol.upper()})
    if strategy:
        settings = settings.model_copy(update={'strategy_name': strategy})

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
    bt = Backtester(strat)
    result = bt.run(df, settings.primary_symbol)

    console.print(result.summary())

    if chart:
        fig = plot_backtest(result)
        if save_html:
            save_html_fn(fig, save_html)
            console.print(f'[green]Chart saved → {save_html}[/]')
        else:
            show(fig)


# ──────────────────────────────────────────────────────────────────────────────
# analyse command
# ──────────────────────────────────────────────────────────────────────────────


@main.command()
@click.option('--symbol', default=None, help='Symbol to analyse.')
@click.option('--bars', default=300, show_default=True, help='Number of bars.')
@click.option('--chart/--no-chart', default=True, show_default=True, help='Open chart.')
def analyse(symbol: str | None, bars: int, chart: bool) -> None:
    """Fetch live data, generate a signal, and optionally show a chart."""
    from trading_bot.data import get_data_provider
    from trading_bot.strategies import get_strategy
    from trading_bot.visualization.charts import plot_ohlcv, show

    settings = get_settings()
    if symbol:
        settings = settings.model_copy(update={'primary_symbol': symbol.upper()})

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
        show(fig)


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
