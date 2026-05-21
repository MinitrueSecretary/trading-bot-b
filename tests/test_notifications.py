"""Tests for the Discord notifier (no actual HTTP calls)."""

from unittest.mock import MagicMock, patch

import pytest

from trading_bot.execution.base_executor import OrderResult, OrderStatus
from trading_bot.notifications.discord import DiscordNotifier
from trading_bot.strategies.base_strategy import Signal, SignalType


@pytest.fixture
def notifier_no_webhook() -> DiscordNotifier:
    """Return a notifier without a webhook (no-op mode)."""
    return DiscordNotifier(webhook_url=None)


@pytest.fixture
def notifier_with_webhook() -> DiscordNotifier:
    """Return a notifier with a fake webhook URL."""
    return DiscordNotifier(webhook_url='https://discord.com/api/webhooks/test/fake')


@pytest.fixture
def buy_signal() -> Signal:
    """Return a sample BUY signal."""
    return Signal(
        signal_type=SignalType.BUY,
        symbol='EURUSD',
        entry_price=1.1000,
        stop_loss=1.0970,
        take_profit=1.1060,
        strength=0.8,
        reason='Test signal',
    )


@pytest.fixture
def order_result() -> OrderResult:
    """Return a sample successful OrderResult."""
    return OrderResult(
        success=True,
        order_id='abc123',
        symbol='EURUSD',
        signal_type=SignalType.BUY,
        price=1.1001,
        volume=0.1,
        stop_loss=1.0971,
        take_profit=1.1061,
        status=OrderStatus.FILLED,
        message='Filled',
    )


class TestDiscordNotifier:
    """Tests for :class:`DiscordNotifier`."""

    def test_no_webhook_does_not_raise(
        self, notifier_no_webhook: DiscordNotifier, buy_signal: Signal
    ) -> None:
        """No-op mode should not raise any exceptions."""
        notifier_no_webhook.send_signal(buy_signal)
        notifier_no_webhook.send_info('Title', 'Body')
        notifier_no_webhook.send_error('Error', 'Details')

    def test_send_signal_calls_http(
        self, notifier_with_webhook: DiscordNotifier, buy_signal: Signal
    ) -> None:
        """send_signal should make an HTTP POST when webhook is set."""
        with patch('trading_bot.notifications.discord.httpx.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_response = MagicMock()
            mock_client.post.return_value = mock_response

            notifier_with_webhook.send_signal(buy_signal)

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            assert 'json' in call_kwargs.kwargs

    def test_send_order_result_calls_http(
        self,
        notifier_with_webhook: DiscordNotifier,
        order_result: OrderResult,
    ) -> None:
        """send_order_result should POST when webhook is configured."""
        with patch('trading_bot.notifications.discord.httpx.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = MagicMock()

            notifier_with_webhook.send_order_result(order_result)
            mock_client.post.assert_called_once()
