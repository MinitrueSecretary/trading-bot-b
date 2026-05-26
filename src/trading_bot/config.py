"""Application configuration via pydantic-settings and .env file."""

import sys
from enum import StrEnum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Timeframe(StrEnum):
    """Supported OHLCV timeframes."""

    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    M30 = '30m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'
    W1 = '1wk'


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file.

    All values can be overridden via a ``.env`` file in the project root or
    by setting the matching environment variable (prefixed with ``BOT_``).
    """

    model_config = SettingsConfigDict(
        env_prefix='BOT_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    # --- General ---
    log_level: str = Field(default='INFO', description='Logging level')
    dry_run: bool = Field(default=True, description='Paper-trade instead of live orders')

    # --- Symbols ---
    symbols: list[str] = Field(
        default=['EURUSD', 'GBPUSD', 'USDJPY'],
        description='Forex pairs to track (MT5 format, no "=X" suffix)',
    )
    primary_symbol: str = Field(default='EURUSD', description='Default symbol for analysis')

    # --- Timeframe ---
    timeframe: Timeframe = Field(default=Timeframe.H1, description='Primary OHLCV timeframe')

    # --- Strategy ---
    strategy_name: str = Field(default='sma_crossover', description='Active strategy')
    sma_fast: int = Field(default=20, gt=1, description='Fast SMA period')
    sma_slow: int = Field(default=50, gt=1, description='Slow SMA period')
    rsi_period: int = Field(default=14, gt=1, description='RSI period')
    rsi_overbought: float = Field(default=70.0, description='RSI overbought threshold')
    rsi_oversold: float = Field(default=30.0, description='RSI oversold threshold')
    macd_fast: int = Field(default=12, description='MACD fast EMA period')
    macd_slow: int = Field(default=26, description='MACD slow EMA period')
    macd_signal: int = Field(default=9, description='MACD signal period')
    stoch_rsi_period: int = Field(default=14, gt=1, description='StochRSI RSI period')
    stoch_rsi_k: int = Field(default=3, gt=0, description='StochRSI K smoothing period')
    stoch_rsi_d: int = Field(default=3, gt=0, description='StochRSI D smoothing period')
    rsi_divergence_lookback: int = Field(
        default=50, gt=1, description='Number of recent bars to search for swing pivot pairs'
    )
    rsi_divergence_swing_window: int = Field(
        default=5, gt=0, description='Bars on each side required to confirm a swing pivot'
    )

    # --- Risk ---
    risk_per_trade_pct: float = Field(
        default=1.0, gt=0, le=100, description='% of balance risked per trade'
    )
    max_open_trades: int = Field(default=3, description='Max concurrent open positions')
    stop_loss_pips: float = Field(default=30.0, description='Default stop-loss in pips')
    take_profit_pips: float = Field(default=60.0, description='Default take-profit in pips')

    # --- MT5 (Windows only) ---
    mt5_login: int | None = Field(default=None, description='MT5 account login')
    mt5_password: str | None = Field(default=None, description='MT5 account password')
    mt5_server: str | None = Field(default=None, description='MT5 broker server')
    mt5_magic: int = Field(default=123456, description='MT5 magic number for order identification')
    mt5_symbol_suffix: str = Field(
        default='m',
        description='Suffix appended to symbol names when calling the MT5 API (e.g. "m" → EURUSDm). Set to "" to disable.',
    )

    # --- Backtest output ---
    backtest_save_chart: bool = Field(
        default=True, description='Save HTML chart after a backtest run'
    )
    backtest_save_csv: bool = Field(
        default=True, description='Save trades CSV after a backtest run'
    )

    # --- Discord ---
    discord_webhook_url: str | None = Field(
        default=None, description='Discord webhook URL for alerts'
    )

    @field_validator('symbols', mode='before')
    @classmethod
    def parse_symbols(cls, v: object) -> list[str]:
        """Allow comma-separated string from env var."""
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(',')]
        return [s.upper() for s in v]  # type: ignore[union-attr]

    @property
    def is_windows(self) -> bool:
        """True when running on Windows (MT5 available)."""
        return sys.platform == 'win32'

    @property
    def yf_symbol(self) -> str:
        """Convert ``primary_symbol`` to yfinance format (e.g. EURUSD → EURUSD=X)."""
        return f'{self.primary_symbol}=X'

    def yf_symbol_for(self, symbol: str) -> str:
        """Convert any MT5-format symbol to yfinance format."""
        return f'{symbol}=X'

    def mt5_symbol_for(self, symbol: str) -> str:
        """Return the broker-specific MT5 symbol name for *symbol*.

        Appends :attr:`mt5_symbol_suffix` so that internal symbols such as
        ``'EURUSD'`` are translated to the broker variant (e.g. ``'EURUSDm'``)
        before any MT5 API call.  Set ``BOT_MT5_SYMBOL_SUFFIX=""`` in the
        environment to leave symbols unchanged.

        Args:
            symbol: Clean symbol name (e.g. ``'EURUSD'``).

        Returns:
            Symbol with the configured suffix appended (e.g. ``'EURUSDm'``).
        """
        return f'{symbol}{self.mt5_symbol_suffix}'


def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
