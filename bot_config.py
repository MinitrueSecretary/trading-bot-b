"""Unified bot configuration.

Defines the composite strategy and position sizing in one place.
Pass this file to the CLI with:
    trading-bot trade    --config bot_config.py
    trading-bot backtest --config bot_config.py

Both functions are optional — omit either one to fall back to the inline
CLI flags (--strategy, --lot-size, --risk-amount, etc.) or .env defaults.
"""

from trading_bot.config import Settings
from trading_bot.strategies.composite import CombineMode, CompositeStrategy, StrategyEntry
from trading_bot.strategies.macd_strategy import MACDStrategy
from trading_bot.strategies.rsi_strategy import RSIStrategy
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy
from trading_bot.utils.lot_size import BaseLotSizeManager, FixedLotSizeManager


def build_strategy(settings: Settings) -> CompositeStrategy:
    """Build and return the composite strategy.

    Args:
        settings: Application settings forwarded from the CLI.

    Returns:
        Configured :class:`~trading_bot.strategies.composite.CompositeStrategy`.
    """
    return CompositeStrategy(
        entries=[
            # ── SMA Crossover ─────────────────────────────────────────────
            StrategyEntry(
                strategy=SMACrossoverStrategy(
                    settings.model_copy(
                        update={
                            'stop_loss_pips': 30,
                            'take_profit_pips': 60,
                            'sma_fast': 20,
                            'sma_slow': 50,
                        }
                    )
                ),
                symbols=['*'],
                weight=1.0,
            ),
            # ── RSI with trend filter ─────────────────────────────────────
            StrategyEntry(
                strategy=RSIStrategy(
                    settings.model_copy(
                        update={
                            'stop_loss_pips': 25,
                            'take_profit_pips': 50,
                            'rsi_period': 14,
                            'rsi_overbought': 70,
                            'rsi_oversold': 30,
                        }
                    )
                ),
                symbols=['EURUSD', 'GBPUSD'],
                weight=2.0,
            ),
            # ── MACD histogram crossover ──────────────────────────────────
            StrategyEntry(
                strategy=MACDStrategy(
                    settings.model_copy(
                        update={
                            'stop_loss_pips': 35,
                            'take_profit_pips': 70,
                            'macd_fast': 12,
                            'macd_slow': 26,
                            'macd_signal': 9,
                        }
                    )
                ),
                symbols=['USDJPY', 'EURJPY'],
                weight=1.5,
            ),
            # ── Bollinger Bands mean-reversion (disabled) ─────────────────
            # StrategyEntry(
            #     strategy=BollingerBandsStrategy(
            #         settings.model_copy(update={
            #             'stop_loss_pips': 100,
            #             'take_profit_pips': 200,
            #             'bb_period': 20,
            #             'bb_std_dev': 2.0,
            #         })
            #     ),
            #     symbols=['BTCUSD', 'ETHUSD'],
            #     weight=1.0,
            # ),
            # ── RSI Divergence (disabled) ─────────────────────────────────
            # StrategyEntry(
            #     strategy=RSIDivergenceStrategy(
            #         settings.model_copy(update={
            #             'stop_loss_pips': 30,
            #             'take_profit_pips': 60,
            #             'stoch_rsi_period': 14,
            #             'stoch_rsi_k': 3,
            #             'stoch_rsi_d': 3,
            #             'rsi_divergence_lookback': 50,
            #             'rsi_divergence_swing_window': 5,
            #         })
            #     ),
            #     symbols=['EURUSD'],
            #     weight=1.5,
            # ),
        ],
        mode=CombineMode.MAJORITY_VOTE,
    )


def build_lot_size_manager() -> BaseLotSizeManager:
    """Return the configured lot size manager.

    Returns:
        An instance of any :class:`~trading_bot.utils.lot_size.BaseLotSizeManager`
        subclass.
    """
    # ── Option A: fixed lot size ───────────────────────────────────────────
    return FixedLotSizeManager(lot_size=0.1)

    # ── Option B: fixed dollar risk per trade ──────────────────────────────
    # from trading_bot.utils.lot_size import FixedRiskLotSizeManager
    # return FixedRiskLotSizeManager(risk_amount=100.0, min_lot=0.01)
