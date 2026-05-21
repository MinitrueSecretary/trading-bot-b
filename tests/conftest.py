"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Settings, Timeframe


@pytest.fixture
def settings() -> Settings:
    """Return a default Settings instance with test-friendly values."""
    return Settings(
        dry_run=True,
        symbols=['EURUSD'],
        primary_symbol='EURUSD',
        timeframe=Timeframe.H1,
        strategy_name='sma_crossover',
        sma_fast=5,
        sma_slow=20,
        rsi_period=14,
        stop_loss_pips=30,
        take_profit_pips=60,
        risk_per_trade_pct=1.0,
        max_open_trades=3,
    )


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with 300 bars."""
    rng = np.random.default_rng(42)
    n = 300
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0002, n))
    spread = rng.uniform(0.0001, 0.0005, n)

    index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    df = pd.DataFrame(
        {
            'open': close - spread / 2,
            'high': close + rng.uniform(0.0001, 0.001, n),
            'low': close - rng.uniform(0.0001, 0.001, n),
            'close': close,
            'volume': rng.integers(100, 10000, n).astype(float),
        },
        index=index,
    )
    # Ensure high >= close and low <= close
    df['high'] = df[['high', 'close', 'open']].max(axis=1)
    df['low'] = df[['low', 'close', 'open']].min(axis=1)
    return df


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """Return a DataFrame with a clear uptrend for signal testing."""
    n = 300
    # Create a steady uptrend
    close = np.linspace(1.0900, 1.1200, n)
    # Add noise around the trend
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.0001, n)
    close = close + noise

    index = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    df = pd.DataFrame(
        {
            'open': close - 0.0001,
            'high': close + 0.0005,
            'low': close - 0.0005,
            'close': close,
            'volume': np.ones(n) * 1000.0,
        },
        index=index,
    )
    df['high'] = df[['high', 'close', 'open']].max(axis=1)
    df['low'] = df[['low', 'close', 'open']].min(axis=1)
    return df
