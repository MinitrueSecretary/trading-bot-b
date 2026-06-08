"""Analyser package — registry and factory."""

from trading_bot.analysers.base_analyser import Analysis, BaseAnalyser
from trading_bot.analysers.ema_trend import EMATrendAnalyser
from trading_bot.analysers.support_resistance import SupportResistanceAnalyser
from trading_bot.analysers.swing_trend import SwingTrendAnalyser
from trading_bot.config import Settings

_REGISTRY: dict[str, type[BaseAnalyser]] = {
    'support_resistance': SupportResistanceAnalyser,
    'ema_trend': EMATrendAnalyser,
    'swing_trend': SwingTrendAnalyser,
}


def get_analyser(name: str, settings: Settings) -> BaseAnalyser:
    """Return an instantiated analyser matching *name*.

    Args:
        name: Registered analyser identifier (e.g. ``'support_resistance'``).
        settings: Application settings forwarded to the analyser constructor.

    Returns:
        Instantiated :class:`BaseAnalyser`.

    Raises:
        ValueError: If *name* is not registered.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        valid = ', '.join(_REGISTRY.keys())
        raise ValueError(f'Unknown analyser {name!r}. Valid options: {valid}')
    return cls(settings)  # type: ignore[call-arg]


def list_analysers() -> list[str]:
    """Return all registered analyser names."""
    return list(_REGISTRY.keys())


__all__ = [
    'Analysis',
    'BaseAnalyser',
    'SupportResistanceAnalyser',
    'EMATrendAnalyser',
    'SwingTrendAnalyser',
    'get_analyser',
    'list_analysers',
]
