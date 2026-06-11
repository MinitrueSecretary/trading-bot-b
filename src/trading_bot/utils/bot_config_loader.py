"""Loader for the unified bot_config.py configuration file."""

import importlib.util
from pathlib import Path

from trading_bot.config import Settings
from trading_bot.strategies.composite import CompositeStrategy
from trading_bot.utils.logger import get_logger
from trading_bot.utils.lot_size import BaseLotSizeManager

logger = get_logger(__name__)


def load_bot_config(
    path: str | Path,
    settings: Settings,
) -> tuple[CompositeStrategy | None, BaseLotSizeManager | None]:
    """Load strategy and lot size manager from a Python bot config file.

    The module may expose either or both of the following callables:

    - ``build_strategy(settings: Settings) -> CompositeStrategy``
    - ``build_lot_size_manager() -> BaseLotSizeManager``

    Both are optional.  If a function is absent the corresponding return value
    is ``None`` and the caller falls back to its default behaviour (inline CLI
    flags or settings defaults).

    Args:
        path: Path to the ``.py`` config file.
        settings: Application settings forwarded to ``build_strategy``.

    Returns:
        ``(strategy, lot_size_manager)`` — either may be ``None`` if the
        corresponding function was not defined in the module.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be loaded as a Python module, or if
            a defined function returns the wrong type.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'Bot config not found: {config_path}')

    spec = importlib.util.spec_from_file_location('_bot_config', config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Could not load bot config: {config_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    strategy: CompositeStrategy | None = None
    build_s = getattr(module, 'build_strategy', None)
    if callable(build_s):
        result_s = build_s(settings)
        if not isinstance(result_s, CompositeStrategy):
            raise ValueError(
                f'`build_strategy()` in {config_path.name} must return a CompositeStrategy, '
                f'got {type(result_s).__name__}.'
            )
        strategy = result_s
        logger.info(
            f'[BotConfig] Loaded strategy: CompositeStrategy '
            f'({len(strategy.entries)} entries, mode={strategy.mode})'
        )

    lot_mgr: BaseLotSizeManager | None = None
    build_l = getattr(module, 'build_lot_size_manager', None)
    if callable(build_l):
        result_l = build_l()
        if not isinstance(result_l, BaseLotSizeManager):
            raise ValueError(
                f'`build_lot_size_manager()` in {config_path.name} must return a '
                f'BaseLotSizeManager, got {type(result_l).__name__}.'
            )
        lot_mgr = result_l
        logger.info(f'[BotConfig] Loaded lot size manager: {type(lot_mgr).__name__}')

    return strategy, lot_mgr
