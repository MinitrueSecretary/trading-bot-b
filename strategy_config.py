"""Composite strategy definition.

Edit this file directly to configure your strategy mix.
Pass it to the CLI with:
    trading-bot trade    --strategy-config strategy_config.py
    trading-bot backtest --strategy-config strategy_config.py

The module must expose a ``build(settings: Settings) -> CompositeStrategy`` function.
Settings overrides (stop_loss_pips, sma_fast, etc.) are applied via model_copy()
so only the keys you specify change — everything else inherits from .env / defaults.
"""

from trading_bot.config import Settings
from trading_bot.strategies.composite import CombineMode, CompositeStrategy, StrategyEntry
from trading_bot.strategies.macd_strategy import MACDStrategy
from trading_bot.strategies.rsi_strategy import RSIStrategy
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy


def build(settings: Settings) -> CompositeStrategy:
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
