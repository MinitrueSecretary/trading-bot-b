"""MT5 live order executor — Windows only."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from trading_bot.config import Settings
from trading_bot.execution.base_executor import BaseExecutor, OrderResult, OrderStatus
from trading_bot.strategies.base_strategy import Signal, SignalType
from trading_bot.utils.logger import get_logger
from trading_bot.utils.pip import get_contract_size

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_MT5_COMMENT_MAX = 63  # MT5 hard limit for the comment field


class MT5Executor(BaseExecutor):
    """Sends live orders to MT5.

    Only functional on Windows with the MT5 terminal running and the
    ``MetaTrader5`` Python package installed.

    Args:
        settings: Application settings.
    """

    # MT5 order type constants (defined here to avoid import at module level)
    _ORDER_TYPE_BUY = 0  # mt5.ORDER_TYPE_BUY
    _ORDER_TYPE_SELL = 1  # mt5.ORDER_TYPE_SELL
    _TRADE_ACTION_DEAL = 1  # mt5.TRADE_ACTION_DEAL
    _ORDER_FILLING_IOC = 1  # mt5.ORDER_FILLING_IOC

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mt5: object | None = None

    def initialise(self) -> None:
        """Import and store the MT5 module reference (call once after connect).

        Raises:
            RuntimeError: On non-Windows or if the MetaTrader5 package is missing.
        """
        if sys.platform != 'win32':
            raise RuntimeError('MT5Executor is only available on Windows.')
        try:
            import MetaTrader5 as mt5  # type: ignore[import]  # noqa: N813

            self._mt5 = mt5
        except ImportError as exc:
            raise RuntimeError('MetaTrader5 package not installed.') from exc

    @property
    def name(self) -> str:
        """Return executor name."""
        return 'MT5Live'

    def place_order(self, signal: Signal, volume: float) -> OrderResult:
        """Send a market order to MT5.

        Args:
            signal: Trading signal.
            volume: Lot size (e.g. 0.01 for micro lot).

        Returns:
            :class:`OrderResult` with fill details.
        """
        if self._mt5 is None:
            self.initialise()

        if signal.signal_type == SignalType.HOLD:
            return OrderResult(
                success=False,
                order_id='',
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                price=signal.entry_price,
                volume=volume,
                status=OrderStatus.REJECTED,
                message='HOLD signal — no order placed',
            )

        mt5 = self._mt5
        order_type = (
            self._ORDER_TYPE_BUY if signal.signal_type == SignalType.BUY else self._ORDER_TYPE_SELL
        )
        mt5_symbol = self._settings.mt5_symbol_for(signal.symbol)

        # Guard: skip if a position for this symbol is already open.
        existing = mt5.positions_get(symbol=mt5_symbol)  # type: ignore[union-attr]
        if existing:
            msg = (
                f'Skipped {signal.signal_type.value} {signal.symbol} — '
                f'{len(existing)} open position(s) already exist for {mt5_symbol}'
            )
            logger.info(f'[MT5Executor] {msg}')
            return OrderResult(
                success=False,
                order_id='',
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                price=signal.entry_price,
                volume=volume,
                status=OrderStatus.REJECTED,
                message=msg,
            )

        request: dict[str, object] = {
            'action': self._TRADE_ACTION_DEAL,
            'symbol': mt5_symbol,
            'volume': volume,
            'type': order_type,
            'price': signal.entry_price,
            'sl': signal.stop_loss,
            'tp': signal.take_profit,
            'deviation': 20,
            'magic': self._settings.mt5_magic,
            'comment': 'trading-bot-b open',
            'type_filling': self._ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)  # type: ignore[union-attr]

        if result is None or result.retcode != 10009:  # type: ignore[union-attr]
            error = mt5.last_error()  # type: ignore[union-attr]
            retcode = getattr(result, 'retcode', 'None')
            msg = f'Order rejected: retcode={retcode} | MT5 error={error}'
            logger.error(f'[MT5Executor] {msg}')
            return OrderResult(
                success=False,
                order_id='',
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                price=signal.entry_price,
                volume=volume,
                status=OrderStatus.REJECTED,
                message=msg,
            )

        order_id = str(result.order)  # type: ignore[union-attr]
        fill_price = float(result.price)  # type: ignore[union-attr]
        contract_size = get_contract_size(signal.symbol)
        logger.info(
            f'[MT5Executor] Order filled: #{order_id} {signal.signal_type.value} '
            f'{volume} {signal.symbol} @ {fill_price:.5f} '
            f'(contract_size={contract_size:,.0f})'
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            price=fill_price,
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=OrderStatus.FILLED,
            message=f'MT5 order filled @ {fill_price:.5f}',
        )

    def close_position(self, order_id: str, symbol: str) -> OrderResult:
        """Close an open MT5 position by ticket.

        Args:
            order_id: MT5 position ticket (as string).
            symbol: Instrument symbol.

        Returns:
            :class:`OrderResult` for the close operation.
        """
        if self._mt5 is None:
            self.initialise()

        mt5 = self._mt5
        ticket = int(order_id)
        positions = mt5.positions_get(ticket=ticket)  # type: ignore[union-attr]

        if not positions:
            return OrderResult(
                success=False,
                order_id=order_id,
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=0.0,
                volume=0.0,
                status=OrderStatus.REJECTED,
                message=f'Position {order_id} not found in MT5',
            )

        pos = positions[0]
        # To close: flip the order type
        close_type = self._ORDER_TYPE_SELL if pos.type == 0 else self._ORDER_TYPE_BUY
        mt5_symbol = self._settings.mt5_symbol_for(symbol)
        tick = mt5.symbol_info_tick(mt5_symbol)  # type: ignore[union-attr]
        close_price = float(tick.bid if close_type == self._ORDER_TYPE_SELL else tick.ask)

        request: dict[str, object] = {
            'action': self._TRADE_ACTION_DEAL,
            'symbol': mt5_symbol,
            'volume': float(pos.volume),
            'type': close_type,
            'position': ticket,
            'price': close_price,
            'deviation': 20,
            'magic': self._settings.mt5_magic,
            'comment': 'trading-bot-b close',
            'type_filling': self._ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)  # type: ignore[union-attr]

        if result is None or result.retcode != 10009:  # type: ignore[union-attr]
            msg = f'Close rejected: retcode={getattr(result, "retcode", "None")}'
            logger.error(f'[MT5Executor] {msg}')
            return OrderResult(
                success=False,
                order_id=order_id,
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=close_price,
                volume=float(pos.volume),
                status=OrderStatus.REJECTED,
                message=msg,
            )

        logger.info(f'[MT5Executor] Position {order_id} closed @ {close_price:.5f}')
        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            price=close_price,
            volume=float(pos.volume),
            status=OrderStatus.FILLED,
            message=f'Closed @ {close_price:.5f}',
        )
