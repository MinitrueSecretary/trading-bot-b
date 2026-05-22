# trading-bot-b

A cross-platform Forex trading bot built with Python 3.12. On **macOS / Linux** it functions as a full algorithm evaluation and visualisation environment using Yahoo Finance as a price feed. On **Windows** it connects to a live MetaTrader 5 terminal and can send real orders based on strategy signals.

---

## Features

- 📊 **Live data** via MetaTrader 5 (Windows) or Yahoo Finance (all platforms)
- 🧠 **Three built-in strategies** — SMA Crossover, RSI, MACD — each swappable at runtime
- 📈 **Interactive charts** — candlestick, volume, RSI panel, equity curve (Plotly)
- 🔁 **Walk-forward backtester** with Sharpe ratio, max drawdown, profit factor, and per-trade P&L
- 💸 **Paper trading** — full in-memory simulation with SL/TP tracking and trade history
- 🚀 **Live order execution** — MT5 market orders with SL/TP (Windows + `dry_run=false`)
- 🔔 **Discord notifications** — rich embeds on every signal and order event
- ⚙️ **Config-driven** — all parameters from `.env` or environment variables, no code changes needed

---

## Architecture

```
trading-bot-b/
├── src/trading_bot/
│   ├── config.py                   # Pydantic-Settings: typed config from .env
│   ├── bot.py                      # Orchestrator: wires all subsystems together
│   ├── cli.py                      # Click CLI entry point
│   │
│   ├── data/                       # Price feed layer
│   │   ├── base_provider.py        # Abstract interface (get_ohlcv, get_current_price)
│   │   ├── yfinance_provider.py    # Yahoo Finance — cross-platform, no account
│   │   └── mt5_provider.py         # MetaTrader 5 — Windows only
│   │
│   ├── strategies/                 # Signal generation layer
│   │   ├── base_strategy.py        # Signal dataclass + BaseStrategy ABC
│   │   ├── sma_crossover.py        # Golden / death cross (fast vs slow SMA)
│   │   ├── rsi_strategy.py         # RSI extremes + SMA200 trend filter
│   │   └── macd_strategy.py        # MACD histogram zero-line crossover
│   │
│   ├── execution/                  # Order execution layer
│   │   ├── base_executor.py        # OrderResult dataclass + BaseExecutor ABC
│   │   ├── paper_executor.py       # In-memory simulation with SL/TP auto-close
│   │   └── mt5_executor.py         # Live MT5 order_send — Windows only
│   │
│   ├── backtesting/
│   │   └── backtester.py           # Bar-by-bar walk-forward engine + statistics
│   │
│   ├── visualization/
│   │   └── charts.py               # Plotly: price chart, backtest equity curve
│   │
│   ├── notifications/
│   │   └── discord.py              # Discord webhook notifier (rich embeds)
│   │
│   └── utils/
│       └── logger.py               # Rich-formatted logging
│
├── tests/                          # pytest test suite
├── main.py                         # `uv run python main.py` entry point
├── pyproject.toml                  # Project metadata and dependencies
├── .env.example                    # All supported config keys with defaults
└── .gitignore
```

### Data flow

```
.env / environment variables
        │
        ▼
    Settings (config.py)
        │
        ├──► DataProvider  (YFinance or MT5)
        │         │  OHLCV DataFrame
        │         ▼
        ├──► Strategy      (SMA / RSI / MACD)
        │         │  Signal (BUY / SELL / HOLD)
        │         ▼
        ├──► Executor      (Paper or MT5Live)
        │         │  OrderResult
        │         ▼
        └──► DiscordNotifier → webhook embed
```

### Platform matrix

| Platform | Data provider | Executor (dry_run=true) | Executor (dry_run=false) |
|---|---|---|---|
| macOS / Linux | YFinance | PaperExecutor | PaperExecutor (fallback) |
| Windows | MT5Provider | PaperExecutor | MT5Executor (live orders) |

The correct provider and executor are selected automatically at runtime — no code changes are needed when switching environments.

---

## Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Homebrew: `brew install python@3.12` |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| MetaTrader 5 terminal | Windows only — download from your broker |

### 1 — Clone and create the environment

```bash
git clone <repo-url>
cd trading-bot-b

uv venv --python 3.12
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2 — Install dependencies

```bash
uv pip install -e ".[dev]"
```

To install the MT5 package on Windows:

```bash
uv pip install -e ".[dev,windows]"
```

### 3 — Configure

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Key settings in `.env`:

```dotenv
# Strategy to run: sma_crossover | rsi | macd
BOT_STRATEGY_NAME=sma_crossover

# Symbols to watch (comma-separated, MT5 format)
BOT_SYMBOLS=EURUSD,GBPUSD,USDJPY

# Timeframe: 1m 5m 15m 30m 1h 4h 1d 1wk
BOT_TIMEFRAME=1h

# false = send real orders via MT5 (Windows only)
BOT_DRY_RUN=true

# Discord webhook for trade alerts (optional)
BOT_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# MT5 credentials (Windows only)
BOT_MT5_LOGIN=12345678
BOT_MT5_PASSWORD=your_password
BOT_MT5_SERVER=YourBroker-Server
```

---

## Usage

All commands are available through the CLI:

```
uv run python main.py [COMMAND] [OPTIONS]
```

### Analyse — fetch live data and generate a signal

```bash
uv run python main.py analyse
uv run python main.py analyse --symbol GBPUSD
uv run python main.py analyse --symbol EURUSD --bars 500 --no-chart
```

Opens an interactive Plotly chart with SMA overlays, RSI panel, and volume. Prints the current signal (BUY / SELL / HOLD) with entry, SL, and TP levels to the terminal.

### Backtest — evaluate a strategy on historical data

```bash
uv run python main.py backtest
uv run python main.py backtest --strategy rsi --symbol USDJPY --bars 2000
uv run python main.py backtest --strategy macd --save-html results.html
```

Prints a statistics table and opens an equity curve chart:

```
════════════════════════════════════════════════════════
  Backtest: sma_crossover on EURUSD
  Period : 2022-01-03 → 2025-12-31
────────────────────────────────────────────────────────
  Initial Balance :  $10,000.00
  Final Balance   :  $11,240.00
  Net Return      :     +12.40%
  Max Drawdown    :       8.20%
  Sharpe Ratio    :        1.14
────────────────────────────────────────────────────────
  Total Trades    :         142
  Win Rate        :      54.2%
  Avg Win         :     +82.50
  Avg Loss        :     -61.30
  Profit Factor   :        1.42
════════════════════════════════════════════════════════
```

### Trade — run the live / paper trading loop

```bash
# Paper trading (default)
uv run python main.py trade

# Override strategy without editing .env
uv run python main.py trade --strategy rsi

# Live MT5 orders (Windows, dry_run must be false in .env)
uv run python main.py trade --live
```

The loop scans all configured symbols every 60 seconds, generates signals, places orders (paper or live), and sends Discord notifications.

### List strategies

```bash
uv run python main.py list-strategies
```

---

## Strategies

### SMA Crossover (`sma_crossover`)

Triggers on a **golden cross** (fast SMA crosses above slow SMA → BUY) or **death cross** (fast crosses below slow → SELL). Simple, robust, and easy to tune.

| Parameter | Default | Description |
|---|---|---|
| `BOT_SMA_FAST` | `20` | Fast SMA period |
| `BOT_SMA_SLOW` | `50` | Slow SMA period |

### RSI (`rsi`)

Enters when RSI crosses back out of an extreme zone, filtered by a 200-SMA trend direction. Only takes BUY signals in uptrends and SELL signals in downtrends.

| Parameter | Default | Description |
|---|---|---|
| `BOT_RSI_PERIOD` | `14` | RSI calculation period |
| `BOT_RSI_OVERBOUGHT` | `70` | Level to trigger SELL |
| `BOT_RSI_OVERSOLD` | `30` | Level to trigger BUY |

### MACD (`macd`)

Generates a signal when the MACD histogram crosses the zero line. BUY on positive cross, SELL on negative cross.

| Parameter | Default | Description |
|---|---|---|
| `BOT_MACD_FAST` | `12` | Fast EMA period |
| `BOT_MACD_SLOW` | `26` | Slow EMA period |
| `BOT_MACD_SIGNAL` | `9` | Signal line period |

---

## Risk Management

| Parameter | Default | Description |
|---|---|---|
| `BOT_RISK_PER_TRADE_PCT` | `1.0` | % of balance risked per trade |
| `BOT_MAX_OPEN_TRADES` | `3` | Max concurrent open positions |
| `BOT_STOP_LOSS_PIPS` | `30` | Default stop-loss in pips |
| `BOT_TAKE_PROFIT_PIPS` | `60` | Default take-profit in pips (2:1 R:R) |

Lot size is calculated automatically from `RISK_PER_TRADE_PCT` and `STOP_LOSS_PIPS`. For live MT5 trading, always verify the calculated lot size is appropriate for your account size and broker's minimum.

---

## Development

### Run tests

```bash
pytest
pytest --cov=trading_bot          # with coverage report
pytest tests/test_strategies.py   # single file
```

### Lint and format

```bash
ruff check . --fix
ruff format .
```

### Type checking

```bash
mypy .
```

### Adding a new strategy

1. Create `src/trading_bot/strategies/my_strategy.py` extending `BaseStrategy`
2. Implement `name`, `required_bars()`, and `generate_signal()` 
3. Register it in `src/trading_bot/strategies/__init__.py`:

```python
from trading_bot.strategies.my_strategy import MyStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {
    ...
    'my_strategy': MyStrategy,
}
```

4. Set `BOT_STRATEGY_NAME=my_strategy` in `.env`

---

## Discord Notifications

Set `BOT_DISCORD_WEBHOOK_URL` in `.env` to receive rich embed alerts:

- 🟢 **BUY signal** — entry, SL, TP, signal strength, reason
- 🔴 **SELL signal** — same fields
- ✅ **Order filled** — order ID, fill price, volume
- ⚠️ **Order rejected** — reason
- ℹ️ **Bot started / stopped**
- 🚨 **Errors** — symbol processing failures

To create a webhook: Discord server → channel settings → **Integrations** → **Webhooks** → **New Webhook** → copy URL.

---

## Licence

MIT
