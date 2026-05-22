"""Tests for :func:`trading_bot.utils.pip.get_pip_size`."""

import pytest

from trading_bot.utils.pip import get_pip_size


class TestGetPipSize:
    """Unit tests for get_pip_size covering all instrument categories."""

    # ── Standard forex ────────────────────────────────────────────────────────

    @pytest.mark.parametrize('symbol', ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCHF', 'USDCAD'])
    def test_standard_forex_returns_0001(self, symbol: str) -> None:
        """Standard 4-decimal forex pairs return 0.0001."""
        assert get_pip_size(symbol) == 0.0001

    # ── JPY pairs ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize('symbol', ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY'])
    def test_jpy_pairs_return_001(self, symbol: str) -> None:
        """JPY pairs (2-decimal) return 0.01."""
        assert get_pip_size(symbol) == 0.01

    # ── Metals ────────────────────────────────────────────────────────────────

    def test_xauusd_gold(self) -> None:
        """Gold (XAUUSD) returns 0.01."""
        assert get_pip_size('XAUUSD') == 0.01

    def test_xagusd_silver(self) -> None:
        """Silver (XAGUSD) returns 0.001."""
        assert get_pip_size('XAGUSD') == 0.001

    def test_xptusd_platinum(self) -> None:
        """Platinum (XPTUSD) returns 0.01."""
        assert get_pip_size('XPTUSD') == 0.01

    def test_xpdusd_palladium(self) -> None:
        """Palladium (XPDUSD) returns 0.01."""
        assert get_pip_size('XPDUSD') == 0.01

    # ── Crypto ────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize('symbol', ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'BNBUSD', 'SOLUSD'])
    def test_crypto_returns_1(self, symbol: str) -> None:
        """Crypto pairs return 1.0 (whole-dollar pip)."""
        assert get_pip_size(symbol) == 1.0

    # ── Case insensitivity ────────────────────────────────────────────────────

    def test_case_insensitive_lower(self) -> None:
        """Lowercase symbols are accepted."""
        assert get_pip_size('eurusd') == 0.0001

    def test_case_insensitive_mixed(self) -> None:
        """Mixed-case symbols are accepted."""
        assert get_pip_size('btcUsd') == 1.0

    def test_case_insensitive_jpy(self) -> None:
        """Lowercase JPY pair is accepted."""
        assert get_pip_size('usdjpy') == 0.01

    def test_case_insensitive_xauusd(self) -> None:
        """Lowercase XAUUSD is accepted."""
        assert get_pip_size('xauusd') == 0.01

    # ── Return type ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize('symbol', ['EURUSD', 'USDJPY', 'XAUUSD', 'XAGUSD', 'BTCUSD'])
    def test_return_type_is_float(self, symbol: str) -> None:
        """get_pip_size always returns a float."""
        result = get_pip_size(symbol)
        assert isinstance(result, float)
