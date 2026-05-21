"""Entry point for the trading bot.

Usage (via uv):
    uv run python main.py [COMMAND] [OPTIONS]

Commands:
    trade            Start live / paper trading loop
    backtest         Run a strategy backtest
    analyse          Fetch live data and show a signal + chart
    list-strategies  Show available strategies
"""

from trading_bot.cli import main

if __name__ == '__main__':
    main()
