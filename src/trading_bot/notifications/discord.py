"""Discord webhook notifier for trade alerts and status updates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from trading_bot.execution.base_executor import OrderResult
from trading_bot.strategies.base_strategy import Signal, SignalType
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)

_COLOUR_BUY = 3066993  # green
_COLOUR_SELL = 15158332  # red
_COLOUR_INFO = 3447003  # blue
_COLOUR_WARN = 15105570  # orange


class DiscordNotifier:
    """Sends rich embed messages to a Discord channel via webhook.

    Args:
        webhook_url: Full Discord webhook URL.  If ``None`` the notifier
            operates in a no-op mode (logs instead of sending).
    """

    def __init__(self, webhook_url: str | None) -> None:
        self._url = webhook_url
        self._enabled = webhook_url is not None

    def _send(self, payload: dict[str, Any]) -> None:
        """POST a JSON payload to the webhook URL.

        Args:
            payload: Discord message payload dict.
        """
        if not self._enabled or not self._url:
            logger.debug(f"[Discord] Skipped (no webhook): {payload}")
            return

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self._url, json=payload)
                resp.raise_for_status()
                logger.debug(f"[Discord] Message sent (status {resp.status_code})")
        except httpx.HTTPError as exc:
            logger.error(f"[Discord] Failed to send message: {exc}")

    def _embed(
        self,
        title: str,
        description: str,
        colour: int,
        fields: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a Discord embed payload.

        Args:
            title: Embed title.
            description: Embed body text.
            colour: Embed left-border colour (decimal int).
            fields: Optional list of ``{name, value, inline}`` dicts.

        Returns:
            Discord webhook payload dict.
        """
        embed: dict[str, Any] = {
            "title": title,
            "description": description,
            "color": colour,
            "timestamp": datetime.now(UTC).isoformat(),
            "footer": {"text": "trading-bot-b"},
        }
        if fields:
            embed["fields"] = fields

        return {"embeds": [embed]}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_signal(self, signal: Signal) -> None:
        """Notify when a new trade signal is generated.

        Args:
            signal: The trading signal.
        """
        colour = _COLOUR_BUY if signal.signal_type == SignalType.BUY else _COLOUR_SELL
        emoji = "🟢" if signal.signal_type == SignalType.BUY else "🔴"

        payload = self._embed(
            title=f"{emoji} {signal.signal_type.value} Signal — {signal.symbol}",
            description=signal.reason,
            colour=colour,
            fields=[
                {"name": "Entry", "value": f"`{signal.entry_price:.5f}`", "inline": True},
                {"name": "Stop Loss", "value": f"`{signal.stop_loss:.5f}`", "inline": True},
                {"name": "Take Profit", "value": f"`{signal.take_profit:.5f}`", "inline": True},
                {"name": "Strength", "value": f"`{signal.strength:.2f}`", "inline": True},
            ],
        )
        logger.info(f"[Discord] Sending signal: {signal.signal_type.value} {signal.symbol}")
        self._send(payload)

    def send_order_result(self, result: OrderResult) -> None:
        """Notify when an order is placed or rejected.

        Args:
            result: The :class:`~trading_bot.execution.base_executor.OrderResult`.
        """
        if result.success:
            colour = _COLOUR_BUY if result.signal_type == SignalType.BUY else _COLOUR_SELL
            emoji = "✅"
        else:
            colour = _COLOUR_WARN
            emoji = "⚠️"

        payload = self._embed(
            title=f"{emoji} Order {result.status.value} — {result.symbol}",
            description=result.message,
            colour=colour,
            fields=[
                {"name": "Order ID", "value": f"`{result.order_id}`", "inline": True},
                {"name": "Direction", "value": f"`{result.signal_type.value}`", "inline": True},
                {"name": "Volume", "value": f"`{result.volume}`", "inline": True},
                {"name": "Fill Price", "value": f"`{result.price:.5f}`", "inline": True},
                {"name": "SL", "value": f"`{result.stop_loss:.5f}`", "inline": True},
                {"name": "TP", "value": f"`{result.take_profit:.5f}`", "inline": True},
            ],
        )
        self._send(payload)

    def send_info(self, title: str, message: str) -> None:
        """Send a generic informational notification.

        Args:
            title: Notification title.
            message: Notification body.
        """
        payload = self._embed(title=f"ℹ️ {title}", description=message, colour=_COLOUR_INFO)
        self._send(payload)

    def send_error(self, title: str, message: str) -> None:
        """Send an error notification.

        Args:
            title: Error title.
            message: Error description.
        """
        payload = self._embed(title=f"🚨 {title}", description=message, colour=_COLOUR_WARN)
        logger.error(f"[Discord] Error notification: {title} — {message}")
        self._send(payload)

    def ping(self) -> bool:
        """Send a test message to verify the webhook is reachable.

        Returns:
            ``True`` if the message was delivered (HTTP 204), ``False`` otherwise.
        """
        if not self._enabled or not self._url:
            logger.warning("[Discord] Ping skipped — no webhook URL configured.")
            return False

        payload = self._embed(
            title="🏓 Ping",
            description="Webhook connection test — all systems go!",
            colour=_COLOUR_INFO,
        )

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self._url, json=payload)
                resp.raise_for_status()
                logger.info(f"[Discord] Ping successful (HTTP {resp.status_code}).")
                return True
        except httpx.HTTPError as exc:
            logger.error(f"[Discord] Ping failed: {exc}")
            return False
