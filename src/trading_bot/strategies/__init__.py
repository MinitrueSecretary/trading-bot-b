"""Strategy package — registry and factory."""

from trading_bot.config import Settings
from trading_bot.strategies.base_strategy import BaseStrategy, Signal, SignalType
from trading_bot.strategies.macd_strategy import MACDStrategy
from trading_bot.strategies.rsi_divergence import RSIDivergenceStrategy
from trading_bot.strategies.rsi_strategy import RSIStrategy
from trading_bot.strategies.sma_crossover import SMACrossoverStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {
    'sma_crossover': SMACrossoverStrategy,
    'rsi': RSIStrategy,
    'macd': MACDStrategy,
    'rsi_divergence': RSIDivergenceStrategy,
}


def get_strategy(settings: Settings) -> BaseStrategy:
    """Return an instantiated strategy matching ``settings.strategy_name``.

    Args:
        settings: Application settings.

    Returns:
        Instantiated :class:`BaseStrategy`.

    Raises:
        ValueError: If the strategy name is not registered.
    """
    cls = _REGISTRY.get(settings.strategy_name)
    if cls is None:
        valid = ', '.join(_REGISTRY.keys())
        raise ValueError(f'Unknown strategy {settings.strategy_name!r}. Valid options: {valid}')
    return cls(settings)


def list_strategies() -> list[str]:
    """Return all registered strategy names."""
    return list(_REGISTRY.keys())


__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType',
    'SMACrossoverStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'RSIDivergenceStrategy',
    'get_strategy',
    'list_strategies',
]
