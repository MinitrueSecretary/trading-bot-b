"""Core bot orchestrator — wires data → strategy → execution → notification."""

import time
from datetime import UTC, datetime

from trading_bot.config import Settings
from trading_bot.data import get_data_provider
from trading_bot.execution import get_executor
from trading_bot.execution.paper_executor import PaperExecutor
from trading_bot.notifications.discord import DiscordNotifier
from trading_bot.strategies import get_strategy
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)


class TradingBot:
    """Main trading bot that integrates all subsystems.

    On construction the appropriate data provider, strategy, executor, and
    Discord notifier are instantiated based on :class:`~trading_bot.config.Settings`.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = get_data_provider(settings)
        self._strategy = get_strategy(settings)
        self._executor = get_executor(settings)
        self._notifier = DiscordNotifier(settings.discord_webhook_url)
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect all components and start the main loop."""
        logger.info(
            f'[Bot] Starting | provider={self._provider.name} '
            f'strategy={self._strategy.name} '
            f'executor={self._executor.name} '
            f'dry_run={self._settings.dry_run}'
        )

        if not self._provider.connect():
            logger.error('[Bot] Failed to connect data provider — aborting.')
            return

        self._notifier.send_info(
            'Bot started',
            f'Strategy: **{self._strategy.name}** | '
            f'Symbols: {", ".join(self._settings.symbols)} | '
            f'Dry run: {self._settings.dry_run}',
        )

        self._running = True
        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info('[Bot] Interrupted by user — shutting down.')
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully disconnect all components."""
        self._running = False
        self._provider.disconnect()
        self._notifier.send_info('Bot stopped', 'Trading bot has shut down.')
        logger.info('[Bot] Stopped.')

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Continuously fetch data, generate signals, and execute orders."""
        poll_interval_s = 60  # seconds between scans

        while self._running:
            scan_start = datetime.now(UTC)
            logger.info(f'[Bot] Scanning {len(self._settings.symbols)} symbol(s)...')

            for symbol in self._settings.symbols:
                try:
                    self._process_symbol(symbol)
                except Exception as exc:
                    logger.error(f'[Bot] Error processing {symbol}: {exc}', exc_info=True)
                    self._notifier.send_error(f'Error on {symbol}', str(exc))

            elapsed = (datetime.now(UTC) - scan_start).total_seconds()
            sleep_s = max(0, poll_interval_s - elapsed)
            logger.info(f'[Bot] Scan complete in {elapsed:.1f}s — sleeping {sleep_s:.0f}s')
            time.sleep(sleep_s)

    def _process_symbol(self, symbol: str) -> None:
        """Fetch data, generate signal, and act on it for one symbol.

        Args:
            symbol: Instrument to process.
        """
        yf_symbol = self._settings.yf_symbol_for(symbol) if not self._settings.is_windows else symbol

        df = self._provider.get_ohlcv(yf_symbol, self._settings.timeframe)
        signal = self._strategy.generate_signal(df, symbol)

        logger.info(f'[Bot] {symbol} → {signal}')

        if not signal.is_actionable:
            return

        # Check open position limits (paper trader only)
        if isinstance(self._executor, PaperExecutor):
            open_count = len(self._executor.open_positions)
            if open_count >= self._settings.max_open_trades:
                logger.info(
                    f'[Bot] Max open trades reached ({open_count}); skipping {symbol}'
                )
                return

        # Notify Discord about the signal
        self._notifier.send_signal(signal)

        # Calculate lot size via fixed risk
        volume = self._calculate_volume()

        # Place the order
        result = self._executor.place_order(signal, volume)
        logger.info(f'[Bot] Order result: {result}')
        self._notifier.send_order_result(result)

        # Update paper positions with current price on same tick
        if isinstance(self._executor, PaperExecutor):
            current_price = self._provider.get_current_price(yf_symbol)
            closed = self._executor.update_positions(symbol, current_price)
            for close_result in closed:
                logger.info(f'[Bot] Position closed: {close_result}')
                self._notifier.send_order_result(close_result)

    def _calculate_volume(self) -> float:
        """Calculate lot size based on fixed risk percentage.

        Returns:
            Lot size rounded to 2 decimal places (minimum 0.01).
        """
        # For MT5 executor, we'd query account balance; paper has it directly
        balance = self._executor.balance if isinstance(self._executor, PaperExecutor) else 10_000.0

        risk_amount = balance * (self._settings.risk_per_trade_pct / 100)
        # Simplified: $1 per pip per 0.1 lot; 1 pip = $10 per lot
        lot = risk_amount / (self._settings.stop_loss_pips * 10)
        return max(0.01, round(lot, 2))
