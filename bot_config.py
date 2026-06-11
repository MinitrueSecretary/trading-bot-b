"""Unified bot configuration.

Defines the composite strategy and position sizing in one place.
Pass this file to the CLI with:
    trading-bot trade    --config bot_config.py
    trading-bot backtest --config bot_config.py

Both functions are optional — omit either one to fall back to the inline
CLI flags (--strategy, --lot-size, --risk-amount, etc.) or .env defaults.
"""

import trading_bot.strategies as S
import trading_bot.utils.lot_size as lot_size
from trading_bot.config import Settings


def build_strategy(settings: Settings) -> S.CompositeStrategy:
    """Build and return the composite strategy.

    Args:
        settings: Application settings forwarded from the CLI.

    Returns:
        Configured :class:`~trading_bot.strategies.composite.CompositeStrategy`.
    """
    return S.CompositeStrategy(
        entries=[
            # ── SMA Crossover ─────────────────────────────────────────────
            # S.StrategyEntry(
            #     strategy=S.SMACrossoverStrategy(
            #         settings.model_copy(
            #             update={
            #                 'stop_loss_pips': 30,
            #                 'take_profit_pips': 60,
            #                 'sma_fast': 20,
            #                 'sma_slow': 50,
            #             }
            #         )
            #     ),
            #     symbols=['*'],
            #     weight=1.0,
            # ),
            # ── RSI with trend filter ─────────────────────────────────────
            S.StrategyEntry(
                strategy=S.RSIStrategy(
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
            S.StrategyEntry(
                strategy=S.MACDStrategy(
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
            # S.StrategyEntry(
            #     strategy=S.BollingerBandsStrategy(
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
            # S.StrategyEntry(
            #     strategy=S.RSIDivergenceStrategy(
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
        mode=S.CombineMode.MAJORITY_VOTE,
    )


def build_lot_size_manager() -> lot_size.BaseLotSizeManager:
    """Return the configured lot size manager.

    Returns:
        An instance of any :class:`~trading_bot.utils.lot_size.BaseLotSizeManager`
        subclass.
    """
    # ── Option A: fixed lot size ───────────────────────────────────────────
    # return lot_size.FixedLotSizeManager(lot_size=0.1)

    # ── Option B: fixed dollar risk per trade ──────────────────────────────
    return lot_size.FixedRiskLotSizeManager(risk_amount=50.0, min_lot=0.01)
