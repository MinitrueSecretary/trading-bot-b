"""Bollinger Bands mean-reversion strategy."""

import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.utils.logger import get_logger
from trading_bot.utils.tpsl import BaseTPSLManager, BollingerBandsTPSLManager

logger = get_logger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean-reversion strategy.

    Generates signals when price bounces off the outer bands:

    * **BUY**: The previous bar closed *at or below* the lower band and the
      current bar closes *back above* it (bullish bounce).
    * **SELL**: The previous bar closed *at or above* the upper band and the
      current bar closes *back below* it (bearish rejection).

    Stop-loss and take-profit levels default to the opposite Bollinger Band
    via :class:`~trading_bot.utils.tpsl.BollingerBandsTPSLManager` but any
    :class:`~trading_bot.utils.tpsl.BaseTPSLManager` can be injected.

    Args:
        settings: Application settings providing ``bb_period``,
            ``bb_std_dev``, ``bb_min_sl_pips``, ``stop_loss_pips``, and
            ``take_profit_pips``.
        tpsl_manager: Optional override for SL/TP calculation.  When
            *None* a :class:`~trading_bot.utils.tpsl.BollingerBandsTPSLManager`
            is constructed from ``settings``.
    """

    def __init__(
        self,
        settings: Settings,
        tpsl_manager: BaseTPSLManager | None = None,
    ) -> None:
        self._period = settings.bb_period
        self._std_dev = settings.bb_std_dev
        self._tpsl = tpsl_manager or BollingerBandsTPSLManager(
            period=settings.bb_period,
            std_dev=settings.bb_std_dev,
            min_sl_pips=settings.bb_min_sl_pips,
        )

    @property
    def name(self) -> str:
        """Return strategy name."""
        return 'bollinger_bands'

    def required_bars(self) -> int:
        """Require enough bars for the band calculation plus one previous bar."""
        return self._period + 1

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Analyse OHLCV data and emit a Bollinger Band bounce signal.

        Args:
            df: OHLCV DataFrame with at least :meth:`required_bars` rows.
            symbol: Instrument symbol.

        Returns:
            A :class:`Signal` of type BUY, SELL, or HOLD.
        """
        if len(df) < self.required_bars():
            logger.warning(
                f'[BB] Insufficient data for {symbol}: '
                f'{len(df)} bars < {self.required_bars()} required'
            )
            return Signal(SignalType.HOLD, symbol, 0.0, reason='Insufficient data')

        df = df.copy()
        bbands = ta.bbands(
            df['close'],
            length=self._period,
            lower_std=self._std_dev,
            upper_std=self._std_dev,
        )
        # Detect column names by prefix to stay robust across pandas-ta versions.
        cols = bbands.columns.tolist()
        lower_col = next((c for c in cols if c.startswith('BBL_')), None)
        middle_col = next((c for c in cols if c.startswith('BBM_')), None)
        upper_col = next((c for c in cols if c.startswith('BBU_')), None)
        if lower_col is None or middle_col is None or upper_col is None:
            return Signal(
                SignalType.HOLD,
                symbol,
                0.0,
                reason=f'Unexpected bbands columns: {cols}',
            )

        df['bb_lower'] = bbands[lower_col]
        df['bb_middle'] = bbands[middle_col]
        df['bb_upper'] = bbands[upper_col]

        # Feed the full pre-dropna df to the TPSL manager so it has enough
        # rows to recompute the Bollinger Bands (period rows required).
        if isinstance(self._tpsl, BollingerBandsTPSLManager):
            self._tpsl.set_data(df)

        # Rebind (not inplace) so the TPSL manager's stored reference is
        # unaffected by the row removal.
        df = df.dropna()

        if len(df) < 2:
            return Signal(
                SignalType.HOLD,
                symbol,
                0.0,
                reason='Not enough data after band calculation',
            )

        price = float(df['close'].iloc[-1])
        prev_close = float(df['close'].iloc[-2])
        curr_lower = float(df['bb_lower'].iloc[-1])
        curr_upper = float(df['bb_upper'].iloc[-1])
        curr_middle = float(df['bb_middle'].iloc[-1])
        prev_lower = float(df['bb_lower'].iloc[-2])
        prev_upper = float(df['bb_upper'].iloc[-2])

        band_width = curr_upper - curr_lower

        # ── BUY: bounce off the lower band ───────────────────────────────────
        if prev_close <= prev_lower and price > curr_lower:
            sl = self._tpsl.calculate_sl(SignalType.BUY, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.BUY, price, symbol)
            percent_b = (price - curr_lower) / band_width if band_width > 0 else 0.5
            strength = max(0.0, min(1.0, 1.0 - percent_b))
            reason = (
                f'Lower band bounce: prev_close={prev_close:.5f} <= '
                f'lower={prev_lower:.5f}, now price={price:.5f} > lower={curr_lower:.5f}'
            )
            logger.info(f'[BB] BUY signal for {symbol} — {reason}')
            return Signal(
                SignalType.BUY,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=strength,
                reason=reason,
                metadata={
                    'bb_lower': curr_lower,
                    'bb_middle': curr_middle,
                    'bb_upper': curr_upper,
                    'band_width': band_width,
                },
            )

        # ── SELL: rejection at the upper band ────────────────────────────────
        if prev_close >= prev_upper and price < curr_upper:
            sl = self._tpsl.calculate_sl(SignalType.SELL, price, symbol)
            tp = self._tpsl.calculate_tp(SignalType.SELL, price, symbol)
            percent_b = (price - curr_lower) / band_width if band_width > 0 else 0.5
            strength = max(0.0, min(1.0, percent_b))
            reason = (
                f'Upper band rejection: prev_close={prev_close:.5f} >= '
                f'upper={prev_upper:.5f}, now price={price:.5f} < upper={curr_upper:.5f}'
            )
            logger.info(f'[BB] SELL signal for {symbol} — {reason}')
            return Signal(
                SignalType.SELL,
                symbol,
                price,
                stop_loss=sl,
                take_profit=tp,
                strength=strength,
                reason=reason,
                metadata={
                    'bb_lower': curr_lower,
                    'bb_middle': curr_middle,
                    'bb_upper': curr_upper,
                    'band_width': band_width,
                },
            )

        return Signal(
            SignalType.HOLD,
            symbol,
            price,
            reason=(
                f'No band touch | lower={curr_lower:.5f} '
                f'middle={curr_middle:.5f} upper={curr_upper:.5f}'
            ),
        )
