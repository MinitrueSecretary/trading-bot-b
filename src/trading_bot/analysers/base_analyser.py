"""Abstract base class and result dataclass for market analysers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go


@dataclass
class Analysis:
    """Result produced by an analyser.

    Unlike a trading signal, an Analysis carries arbitrary structured market
    information (levels, trends, zones, etc.) rather than a directional
    recommendation.

    Attributes:
        symbol: Instrument the analysis applies to.
        data: Arbitrary key-value pairs produced by the analyser.
            Convention: use plain Python scalars or lists of scalars so the
            result can be serialised easily (e.g.
            ``{'support': [1.123, 1.150], 'resistance': [1.200, 1.250]}``).
        notes: Human-readable summary of the most relevant findings.
        metadata: Extra analyser-specific fields not part of the primary data.
    """

    symbol: str
    data: dict[str, object]
    notes: str = ''
    metadata: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f'[Analysis] {self.symbol}']
        for key, val in self.data.items():
            lines.append(f'  {key}: {val}')
        if self.notes:
            lines.append(f'  notes: {self.notes}')
        return '\n'.join(lines)


class BaseAnalyser(ABC):
    """Interface for all market analysers.

    Analysers study OHLCV data and return structured findings without
    necessarily committing to a BUY or SELL direction.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Analyser identifier (e.g. ``'support_resistance'``)."""

    @abstractmethod
    def required_bars(self) -> int:
        """Minimum number of bars required to produce a valid analysis."""

    @abstractmethod
    def analyse(self, df: pd.DataFrame, symbol: str) -> Analysis:
        """Analyse *df* and return an :class:`Analysis`.

        Args:
            df: OHLCV DataFrame (columns: ``open high low close volume``).
                Must have at least as many rows as :meth:`required_bars`.
            symbol: Instrument the data belongs to.

        Returns:
            An :class:`Analysis` containing the findings.
        """

    @abstractmethod
    def plot(self, df: pd.DataFrame, analysis: Analysis, symbol: str) -> 'go.Figure':
        """Build a chart visualising the analysis results.

        Args:
            df: Full OHLCV DataFrame — implementations slice internally for display.
            analysis: The analysis result to visualise.
            symbol: Instrument name used in the chart title.

        Returns:
            A :class:`plotly.graph_objects.Figure` ready to save or display.
        """
