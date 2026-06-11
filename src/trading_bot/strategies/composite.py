"""Composite strategy — combine multiple strategies with configurable voting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.utils.logger import get_logger

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = get_logger(__name__)


class CombineMode(StrEnum):
    """How sub-strategy signals are combined into a single output signal.

    Attributes:
        MAJORITY_VOTE: Direction with more votes (unweighted) wins.
            Ties resolve to HOLD.
        WEIGHTED_VOTE: Direction whose entries sum to a higher weight wins.
            Ties resolve to HOLD.
        FIRST_SIGNAL: First non-HOLD signal from the ordered entry list wins.
        ALL_AGREE: Only signal when every applicable strategy agrees on the
            same direction. A single dissenter produces HOLD.
    """

    MAJORITY_VOTE = 'majority_vote'
    WEIGHTED_VOTE = 'weighted_vote'
    FIRST_SIGNAL = 'first_signal'
    ALL_AGREE = 'all_agree'


@dataclass
class StrategyEntry:
    """One sub-strategy within a :class:`CompositeStrategy`.

    Attributes:
        strategy: Instantiated strategy to run.
        symbols: Instruments this entry is active for. ``['*']`` means all.
        weight: Relative influence in weighted / majority voting (default 1.0).
    """

    strategy: BaseStrategy
    symbols: list[str]
    weight: float = 1.0

    def applies_to(self, symbol: str) -> bool:
        """Return ``True`` if this entry should generate a signal for *symbol*.

        Args:
            symbol: Instrument in MT5 format (e.g. ``'EURUSD'``).

        Returns:
            ``True`` when ``symbols`` contains ``'*'`` or the symbol itself.
        """
        normalised = {s.upper() for s in self.symbols}
        return '*' in normalised or symbol.upper() in normalised


class CompositeStrategy(BaseStrategy):
    """Combines multiple sub-strategies using a configurable voting mechanism.

    Each sub-strategy can be scoped to a subset of symbols.  The composite
    asks every applicable sub-strategy for a signal and merges the results
    according to *mode*.  SL/TP levels come from the highest-strength signal
    in the winning direction.

    Args:
        entries: Ordered list of sub-strategy entries.
        mode: Voting / combination mode (default: :attr:`CombineMode.MAJORITY_VOTE`).

    Raises:
        ValueError: If *entries* is empty.

    Example::

        composite = CompositeStrategy(
            entries=[
                StrategyEntry(SMACrossoverStrategy(settings), symbols=['*']),
                StrategyEntry(RSIStrategy(settings), symbols=['EURUSD', 'GBPUSD'], weight=2.0),
            ],
            mode=CombineMode.WEIGHTED_VOTE,
        )
    """

    def __init__(
        self,
        entries: list[StrategyEntry],
        mode: CombineMode = CombineMode.MAJORITY_VOTE,
    ) -> None:
        if not entries:
            raise ValueError('CompositeStrategy requires at least one StrategyEntry.')
        self._entries = entries
        self._mode = mode

    @property
    def name(self) -> str:
        """Return strategy name."""
        return 'composite'

    @property
    def mode(self) -> CombineMode:
        """Active combination mode."""
        return self._mode

    @property
    def entries(self) -> list[StrategyEntry]:
        """Read-only view of sub-strategy entries."""
        return list(self._entries)

    def required_bars(self) -> int:
        """Return the maximum ``required_bars`` across all entries."""
        return max(e.strategy.required_bars() for e in self._entries)

    # ── Signal generation ──────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Ask every applicable sub-strategy for a signal and combine them.

        Args:
            df: OHLCV DataFrame with at least :meth:`required_bars` rows.
            symbol: Instrument symbol.

        Returns:
            A combined :class:`Signal` — BUY, SELL, or HOLD.
        """
        applicable = [
            (e, e.strategy.generate_signal(df, symbol))
            for e in self._entries
            if e.applies_to(symbol)
        ]

        if not applicable:
            return Signal(
                SignalType.HOLD,
                symbol,
                0.0,
                reason=f'[composite] No strategy configured for {symbol}',
            )

        logger.debug(
            f'[Composite:{self._mode}] {symbol} — '
            + ' | '.join(f'{e.strategy.name}={s.signal_type.value}' for e, s in applicable)
        )

        return self._combine(applicable, symbol)

    def _combine(
        self,
        applicable: list[tuple[StrategyEntry, Signal]],
        symbol: str,
    ) -> Signal:
        """Merge per-entry signals according to :attr:`mode`.

        Args:
            applicable: List of ``(entry, signal)`` pairs for this symbol.
            symbol: Instrument symbol.

        Returns:
            Combined :class:`Signal`.
        """
        if self._mode == CombineMode.FIRST_SIGNAL:
            for entry, sig in applicable:
                if sig.is_actionable:
                    return self._tag(sig, entry.strategy.name, applicable)
            return applicable[0][1]

        if self._mode == CombineMode.ALL_AGREE:
            # HOLDs are treated as abstentions — only non-HOLD signals must agree.
            actionable = [(e, s) for e, s in applicable if s.is_actionable]
            if not actionable:
                return applicable[0][1]
            directions = {s.signal_type for _, s in actionable}
            if len(directions) == 1:
                best_entry, best_sig = max(actionable, key=lambda t: t[1].strength)
                return self._tag(best_sig, best_entry.strategy.name, applicable)
            disagree = [
                f'{e.strategy.name}={s.signal_type.value}' for e, s in applicable if s.is_actionable
            ]
            return Signal(
                SignalType.HOLD,
                symbol,
                0.0,
                reason=f'[composite:all_agree] Actionable strategies disagree: {disagree}',
                metadata={'votes': {e.strategy.name: s.signal_type.value for e, s in applicable}},
            )

        # MAJORITY_VOTE or WEIGHTED_VOTE
        buy_pairs: list[tuple[StrategyEntry, Signal]] = []
        sell_pairs: list[tuple[StrategyEntry, Signal]] = []

        for entry, sig in applicable:
            if sig.signal_type == SignalType.BUY:
                buy_pairs.append((entry, sig))
            elif sig.signal_type == SignalType.SELL:
                sell_pairs.append((entry, sig))

        if not buy_pairs and not sell_pairs:
            return applicable[0][1]

        buy_score = sum(e.weight for e, _ in buy_pairs)
        sell_score = sum(e.weight for e, _ in sell_pairs)
        total = buy_score + sell_score

        if buy_score == sell_score:
            return Signal(
                SignalType.HOLD,
                symbol,
                0.0,
                reason=f'[composite:{self._mode}] Tied vote (buy={buy_score}, sell={sell_score})',
                metadata={
                    'buy_score': buy_score,
                    'sell_score': sell_score,
                    'votes': {e.strategy.name: s.signal_type.value for e, s in applicable},
                },
            )

        winning_pairs = buy_pairs if buy_score > sell_score else sell_pairs
        winning_score = max(buy_score, sell_score)
        best_entry, best_sig = max(winning_pairs, key=lambda t: t[1].strength)

        composite_strength = winning_score / total if total > 0 else 1.0

        combined = Signal(
            best_sig.signal_type,
            symbol,
            best_sig.entry_price,
            stop_loss=best_sig.stop_loss,
            take_profit=best_sig.take_profit,
            strength=composite_strength,
            reason=f'[composite:{self._mode}] {best_sig.reason}',
            metadata={
                **best_sig.metadata,
                'winning_strategy': best_entry.strategy.name,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'votes': {e.strategy.name: s.signal_type.value for e, s in applicable},
            },
        )
        return combined

    @staticmethod
    def _tag(
        sig: Signal, winner_name: str, applicable: list[tuple[StrategyEntry, Signal]]
    ) -> Signal:
        """Return *sig* enriched with composite metadata."""
        return Signal(
            sig.signal_type,
            sig.symbol,
            sig.entry_price,
            stop_loss=sig.stop_loss,
            take_profit=sig.take_profit,
            strength=sig.strength,
            reason=f'[composite] {sig.reason}',
            metadata={
                **sig.metadata,
                'winning_strategy': winner_name,
                'votes': {e.strategy.name: s.signal_type.value for e, s in applicable},
            },
        )

    # ── Visualisation ──────────────────────────────────────────────────────

    def plot_signal(self, df: pd.DataFrame, signal: Signal, symbol: str) -> go.Figure:
        """Delegate to the winning sub-strategy's chart.

        The winning strategy is identified from ``signal.metadata['winning_strategy']``
        when available, falling back to the first applicable entry.

        Args:
            df: Full OHLCV DataFrame.
            signal: Combined signal produced by :meth:`generate_signal`.
            symbol: Instrument symbol.

        Returns:
            A :class:`plotly.graph_objects.Figure` from the winning sub-strategy.
        """
        applicable = [e for e in self._entries if e.applies_to(symbol)]
        if not applicable:
            import plotly.graph_objects as go

            return go.Figure()

        winning_name: str | None = signal.metadata.get('winning_strategy')  # type: ignore[assignment]
        if winning_name:
            entry = next((e for e in applicable if e.strategy.name == winning_name), applicable[0])
        else:
            entry = applicable[0]

        return entry.strategy.plot_signal(df, signal, symbol)


# ── YAML config loader ─────────────────────────────────────────────────────────


def load_composite_strategy(config_path: str | Path, settings: Settings) -> CompositeStrategy:
    """Build a :class:`CompositeStrategy` from a config file.

    Supports two formats detected by file extension:

    **Python** (``.py``):
        The module must expose a ``build(settings: Settings) -> CompositeStrategy``
        function.  The function receives the full ``Settings`` object so it can
        call ``settings.model_copy(update={...})`` for per-entry overrides.

    **YAML** (``.yaml`` / ``.yml``):
        Legacy format.  Schema::

            mode: majority_vote   # majority_vote | weighted_vote | first_signal | all_agree

            strategies:
              - strategy: sma_crossover
                enabled: true          # omit or true to include; false to skip
                symbols: ["*"]         # ["*"] = all symbols; or e.g. ["EURUSD","GBPUSD"]
                weight: 1.0            # relative voting weight (default 1.0)
                params:                # override any BOT_* settings field
                  stop_loss_pips: 30
                  sma_fast: 20

        ``params`` keys must match :class:`~trading_bot.config.Settings` field names.
        Unknown keys are silently ignored (``Settings`` has ``extra='ignore'``).

    Args:
        config_path: Path to the ``.py`` or ``.yaml`` / ``.yml`` file.
        settings: Base settings object forwarded to the loader.

    Returns:
        A fully configured :class:`CompositeStrategy`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file extension is unsupported, an unknown strategy
            name is used, the ``.py`` module has no ``build`` function, or no
            enabled entries remain (YAML only).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Strategy config not found: {path}')

    if path.suffix == '.py':
        return _load_from_py(path, settings)
    if path.suffix in ('.yaml', '.yml'):
        return _load_from_yaml(path, settings)

    raise ValueError(
        f'Unsupported strategy config extension {path.suffix!r}. Use .py or .yaml / .yml.'
    )


def _load_from_py(path: Path, settings: Settings) -> CompositeStrategy:
    """Load a composite strategy from a Python module.

    The module must define ``build(settings: Settings) -> CompositeStrategy``.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location('_strategy_config', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Could not load Python strategy config: {path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    build_fn = getattr(module, 'build', None)
    if build_fn is None or not callable(build_fn):
        raise ValueError(f'{path} must define a callable `build(settings) -> CompositeStrategy`.')

    result = build_fn(settings)
    if not isinstance(result, CompositeStrategy):
        raise ValueError(
            f'`build()` in {path} must return a CompositeStrategy, got {type(result).__name__}.'
        )

    logger.info(
        f'[CompositeConfig] Loaded from {path.name}: {len(result.entries)} entries, mode={result.mode}'
    )
    return result


def _load_from_yaml(path: Path, settings: Settings) -> CompositeStrategy:
    """Load a composite strategy from a YAML config file (legacy format)."""
    from trading_bot.strategies import _REGISTRY  # avoid circular at module level

    with path.open() as fh:
        raw: dict = yaml.safe_load(fh) or {}

    mode = CombineMode(raw.get('mode', CombineMode.MAJORITY_VOTE))

    entries: list[StrategyEntry] = []
    for entry_cfg in raw.get('strategies', []):
        if not entry_cfg.get('enabled', True):
            continue

        strategy_name: str = entry_cfg['strategy']
        if strategy_name not in _REGISTRY:
            valid = ', '.join(_REGISTRY)
            raise ValueError(
                f'Unknown strategy {strategy_name!r} in composite config. Valid options: {valid}'
            )

        symbols: list[str] = entry_cfg.get('symbols', ['*'])
        weight: float = float(entry_cfg.get('weight', 1.0))
        params: dict = entry_cfg.get('params', {})

        entry_settings = settings.model_copy(update=params) if params else settings
        strategy_instance = _REGISTRY[strategy_name](entry_settings)
        entries.append(StrategyEntry(strategy=strategy_instance, symbols=symbols, weight=weight))

        logger.info(f'[CompositeConfig] Loaded {strategy_name} symbols={symbols} weight={weight}')

    if not entries:
        raise ValueError(f'No enabled strategies found in composite config: {path}')

    logger.info(f'[CompositeConfig] mode={mode} entries={len(entries)}')
    return CompositeStrategy(entries=entries, mode=mode)
