"""Vectorised backtesting engine for evaluating trading strategies."""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.strategies.base_strategy import BaseStrategy, SignalType
from trading_bot.utils.logger import get_logger
from trading_bot.utils.lot_size import BaseLotSizeManager, FixedLotSizeManager
from trading_bot.utils.pip import get_pip_size

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Summary statistics from a completed backtest run.

    Attributes:
        symbol: Instrument tested.
        strategy_name: Name of the strategy.
        start_date: First bar date.
        end_date: Last bar date.
        initial_balance: Starting capital.
        final_balance: Ending capital.
        total_return_pct: Percentage return over the test period.
        total_trades: Total number of trades taken.
        winning_trades: Number of profitable trades.
        losing_trades: Number of unprofitable trades.
        win_rate: Percentage of winning trades.
        avg_win: Average profit on winning trades.
        avg_loss: Average loss on losing trades.
        profit_factor: Gross profit / gross loss.
        max_drawdown_pct: Maximum peak-to-trough drawdown.
        sharpe_ratio: Annualised Sharpe ratio (risk-free rate = 0).
        trades: DataFrame of individual trade records.
        equity_curve: Series of equity values over time.
    """

    symbol: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.Series = field(default_factory=pd.Series)  # type: ignore[type-arg]

    def save_trades_csv(
        self,
        path: str | Path | None = None,
        output_dir: str | Path = 'trades',
    ) -> Path:
        """Save the trades DataFrame to a CSV file and return the path.

        If *path* is given it is used directly; otherwise a timestamped file
        is created inside *output_dir* following the same convention as
        :func:`~trading_bot.visualization.charts.save_chart`.

        Args:
            path: Explicit output path (e.g. ``'my_trades.csv'``).  When
                ``None`` an auto-generated name is used.
            output_dir: Folder for auto-generated files (default: ``trades/``
                relative to cwd).  Ignored when *path* is provided.

        Returns:
            :class:`~pathlib.Path` of the saved file.
        """
        if path is not None:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            folder = Path(output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')
            safe_symbol = re.sub(r'[^\w]', '_', self.symbol)
            dest = folder / f'{safe_symbol}_{self.strategy_name}_{ts}.csv'

        self.trades.to_csv(dest, index=False)
        return dest

    def summary(self) -> str:
        """Return a formatted text summary of the backtest results."""
        return (
            f'\n{"=" * 56}\n'
            f'  Backtest: {self.strategy_name} on {self.symbol}\n'
            f'  Period : {self.start_date:%Y-%m-%d} → {self.end_date:%Y-%m-%d}\n'
            f'{"─" * 56}\n'
            f'  Initial Balance : ${self.initial_balance:>10,.2f}\n'
            f'  Final Balance   : ${self.final_balance:>10,.2f}\n'
            f'  Net Return      : {self.total_return_pct:>+10.2f}%\n'
            f'  Max Drawdown    : {self.max_drawdown_pct:>10.2f}%\n'
            f'  Sharpe Ratio    : {self.sharpe_ratio:>10.2f}\n'
            f'{"─" * 56}\n'
            f'  Total Trades    : {self.total_trades:>10}\n'
            f'  Win Rate        : {self.win_rate:>10.1f}%\n'
            f'  Avg Win         : {self.avg_win:>+10.2f}\n'
            f'  Avg Loss        : {self.avg_loss:>+10.2f}\n'
            f'  Profit Factor   : {self.profit_factor:>10.2f}\n'
            f'{"=" * 56}'
        )


class Backtester:
    """Walk-forward backtesting engine.

    Iterates over OHLCV data bar-by-bar, feeds an expanding window to the
    strategy, and simulates order fills at the next bar's open.

    Args:
        strategy: An instantiated :class:`~trading_bot.strategies.base_strategy.BaseStrategy`.
        initial_balance: Starting capital (default $10 000).
        lot_size: Fixed lot size per trade (default 0.1).
        commission_per_lot: Commission in price-units per lot (default 0).
        slippage_pips: Simulated slippage in pips (default 1).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_balance: float = 10_000.0,
        lot_size: float = 0.1,
        commission_per_lot: float = 0.0,
        slippage_pips: float = 1.0,
        lot_size_manager: BaseLotSizeManager | None = None,
    ) -> None:
        self._strategy = strategy
        self._initial_balance = initial_balance
        self._commission = commission_per_lot
        self._slippage = slippage_pips
        self._lot_size_manager = lot_size_manager or FixedLotSizeManager(lot_size)

    def run(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Execute the backtest over *df*.

        Args:
            df: Full OHLCV DataFrame to test on.
            symbol: Instrument symbol (used for pip calculations).

        Returns:
            A :class:`BacktestResult` with full statistics.
        """
        required = self._strategy.required_bars()
        pip_size = get_pip_size(symbol)

        balance = self._initial_balance
        equity_values: list[float] = [balance]
        equity_index: list[object] = [df.index[0]]
        trades: list[dict[str, object]] = []

        in_trade: dict[str, object] | None = None

        logger.info(
            f'[Backtest] Starting {self._strategy.name} on {symbol} '
            f'| {len(df)} bars | initial=${self._initial_balance:,.0f}'
        )

        for i in range(required, len(df) - 1):
            window = df.iloc[: i + 1]
            current_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]  # entry at next bar open

            # ── Check exit conditions for open trade ──────────────────────
            if in_trade is not None:
                direction = in_trade['direction']
                sl = float(in_trade['stop_loss'])  # type: ignore[arg-type]
                tp = float(in_trade['take_profit'])  # type: ignore[arg-type]
                entry = float(in_trade['entry_price'])  # type: ignore[arg-type]
                vol = float(in_trade['volume'])  # type: ignore[arg-type]
                bar_high = float(current_bar['high'])
                bar_low = float(current_bar['low'])

                pnl = 0.0
                exit_price = 0.0
                exit_reason = ''

                if direction == SignalType.BUY:
                    if bar_low <= sl and sl > 0:
                        exit_price = sl
                        exit_reason = 'SL'
                    elif bar_high >= tp and tp > 0:
                        exit_price = tp
                        exit_reason = 'TP'
                    if exit_price:
                        pnl = (exit_price - entry) * vol * 100_000 - self._commission * vol
                elif direction == SignalType.SELL:
                    if bar_high >= sl and sl > 0:
                        exit_price = sl
                        exit_reason = 'SL'
                    elif bar_low <= tp and tp > 0:
                        exit_price = tp
                        exit_reason = 'TP'
                    if exit_price:
                        pnl = (entry - exit_price) * vol * 100_000 - self._commission * vol

                if exit_price:
                    balance += pnl
                    in_trade['exit_price'] = exit_price  # type: ignore[index]
                    in_trade['exit_reason'] = exit_reason  # type: ignore[index]
                    in_trade['pnl'] = pnl  # type: ignore[index]
                    in_trade['exit_index'] = df.index[i]  # type: ignore[index]
                    trades.append(in_trade)  # type: ignore[arg-type]
                    in_trade = None

            # ── Generate signal ───────────────────────────────────────────
            if in_trade is None:
                signal = self._strategy.generate_signal(window, symbol)

                if signal.is_actionable:
                    slippage = self._slippage * pip_size
                    if signal.signal_type == SignalType.BUY:
                        fill_price = float(next_bar['open']) + slippage
                    else:
                        fill_price = float(next_bar['open']) - slippage

                    in_trade = {
                        'symbol': symbol,
                        'direction': signal.signal_type,
                        'entry_price': fill_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'volume': self._lot_size_manager.get_lot_size(
                            balance, fill_price, symbol, signal.stop_loss
                        ),
                        'entry_index': df.index[i + 1],
                        'reason': signal.reason,
                    }

            equity_values.append(balance)
            equity_index.append(df.index[i])

        # ── Force-close any open trade at last bar ────────────────────────
        if in_trade is not None:
            last_price = float(df['close'].iloc[-1])
            direction = in_trade['direction']
            entry = float(in_trade['entry_price'])  # type: ignore[arg-type]
            vol = float(in_trade['volume'])  # type: ignore[arg-type]
            pnl = (
                (last_price - entry) * vol * 100_000
                if direction == SignalType.BUY
                else (entry - last_price) * vol * 100_000
            )
            balance += pnl
            in_trade['exit_price'] = last_price
            in_trade['exit_reason'] = 'End of data'
            in_trade['pnl'] = pnl
            in_trade['exit_index'] = df.index[-1]
            trades.append(in_trade)  # type: ignore[arg-type]

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_series = pd.Series(equity_values, index=equity_index)

        return self._compute_stats(
            symbol=symbol,
            start_date=df.index[0].to_pydatetime(),
            end_date=df.index[-1].to_pydatetime(),
            balance=balance,
            trades_df=trades_df,
            equity_series=equity_series,
        )

    def _compute_stats(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        balance: float,
        trades_df: pd.DataFrame,
        equity_series: pd.Series,  # type: ignore[type-arg]
    ) -> BacktestResult:
        """Compute all summary statistics from raw trade data."""
        total_return_pct = (balance - self._initial_balance) / self._initial_balance * 100

        if trades_df.empty:
            return BacktestResult(
                symbol=symbol,
                strategy_name=self._strategy.name,
                start_date=start_date,
                end_date=end_date,
                initial_balance=self._initial_balance,
                final_balance=balance,
                total_return_pct=total_return_pct,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                trades=trades_df,
                equity_curve=equity_series,
            )

        pnls = trades_df['pnl'].astype(float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        gross_profit = wins.sum() if not wins.empty else 0.0
        gross_loss = abs(losses.sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown from equity curve
        equity_arr = equity_series.values.astype(float)
        rolling_max = np.maximum.accumulate(equity_arr)
        drawdowns = (rolling_max - equity_arr) / rolling_max * 100
        max_drawdown_pct = float(np.max(drawdowns))

        # Sharpe ratio (annualised, zero risk-free rate)
        returns = equity_series.pct_change().dropna()
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        return BacktestResult(
            symbol=symbol,
            strategy_name=self._strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self._initial_balance,
            final_balance=balance,
            total_return_pct=total_return_pct,
            total_trades=len(trades_df),
            winning_trades=int(len(wins)),
            losing_trades=int(len(losses)),
            win_rate=float(len(wins) / len(trades_df) * 100),
            avg_win=float(wins.mean()) if not wins.empty else 0.0,
            avg_loss=float(losses.mean()) if not losses.empty else 0.0,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            trades=trades_df,
            equity_curve=equity_series,
        )
