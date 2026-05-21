"""Tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError

from trading_bot.config import Settings, Timeframe


def test_default_settings() -> None:
    """Default settings should be valid and populated."""
    s = Settings()
    assert s.dry_run is True
    assert s.primary_symbol == 'EURUSD'
    assert s.sma_fast < s.sma_slow
    assert 0 < s.risk_per_trade_pct <= 100


def test_symbols_normalised_to_uppercase() -> None:
    """Symbols should always be upper-cased."""
    s = Settings(symbols=['eurusd', 'gbpusd'])
    assert s.symbols == ['EURUSD', 'GBPUSD']


def test_symbols_from_comma_string() -> None:
    """Comma-separated string symbols should be parsed correctly."""
    s = Settings(symbols='eurusd, gbpusd, usdjpy')  # type: ignore[arg-type]
    assert s.symbols == ['EURUSD', 'GBPUSD', 'USDJPY']


def test_yf_symbol_property() -> None:
    """yf_symbol should append '=X' to the primary symbol."""
    s = Settings(primary_symbol='GBPUSD')
    assert s.yf_symbol == 'GBPUSD=X'


def test_yf_symbol_for() -> None:
    """yf_symbol_for should convert any symbol correctly."""
    s = Settings()
    assert s.yf_symbol_for('USDJPY') == 'USDJPY=X'


def test_invalid_risk_pct() -> None:
    """Risk > 100 should raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(risk_per_trade_pct=101)


def test_timeframe_enum() -> None:
    """Timeframe should accept string values."""
    s = Settings(timeframe='4h')  # type: ignore[arg-type]
    assert s.timeframe == Timeframe.H4
