"""Pip size and contract size utilities for various instrument types."""

# Instrument-type prefix lookup (checked in order; first match wins).
# Add new entries here to extend support without touching strategy code.
_CRYPTO_PREFIXES: tuple[str, ...] = ('BTC', 'ETH', 'LTC', 'XRP', 'BNB', 'SOL', 'ADA', 'DOGE')
_METAL_PIP: dict[str, float] = {
    'XAU': 0.01,  # Gold  — quoted to 2 d.p. (~$0.01 moves)
    'XAG': 0.001,  # Silver — quoted to 3 d.p. (~$0.001 moves)
    'XPT': 0.01,  # Platinum
    'XPD': 0.01,  # Palladium
}

# Contract size (units per standard lot) by instrument prefix.
# Forex default is 100 000 (1 standard lot = 100 000 base units).
_METAL_CONTRACT: dict[str, float] = {
    'XAU': 100.0,  # Gold   — 100 troy oz per lot
    'XAG': 5_000.0,  # Silver — 5 000 troy oz per lot
    'XPT': 100.0,  # Platinum
    'XPD': 100.0,  # Palladium
}


def get_pip_size(symbol: str) -> float:
    """Return the pip size for a given trading symbol.

    Instrument pip sizes:

    - **Crypto** (BTC, ETH, …): ``1.0``  — price moves in whole dollars.
    - **Gold (XAUUSD)**: ``0.01``
    - **Silver (XAGUSD)**: ``0.001``
    - **Other metals (XPT, XPD)**: ``0.01``
    - **JPY pairs**: ``0.01``
    - **Standard forex**: ``0.0001``

    Args:
        symbol: The trading instrument symbol, e.g. ``'EURUSD'``, ``'BTCUSD'``,
            ``'XAUUSD'``, ``'USDJPY'``.

    Returns:
        The pip size as a ``float``.

    Examples:
        >>> get_pip_size('EURUSD')
        0.0001
        >>> get_pip_size('USDJPY')
        0.01
        >>> get_pip_size('XAUUSD')
        0.01
        >>> get_pip_size('XAGUSD')
        0.001
        >>> get_pip_size('BTCUSD')
        1.0
    """
    sym = symbol.upper()

    if any(sym.startswith(prefix) for prefix in _CRYPTO_PREFIXES):
        return 1.0

    for metal, pip in _METAL_PIP.items():
        if metal in sym:
            return pip

    if 'JPY' in sym:
        return 0.01

    return 0.00001


def get_contract_size(symbol: str) -> float:
    """Return the contract size (units per standard lot) for a given symbol.

    Contract sizes by instrument type:

    - **Gold (XAU)**: ``100`` — 100 troy oz per lot
    - **Silver (XAG)**: ``5 000`` — 5 000 troy oz per lot
    - **Platinum / Palladium (XPT, XPD)**: ``100``
    - **Crypto** (BTC, ETH, …): ``1`` — 1 unit per lot
    - **Standard forex / other**: ``100 000``

    Args:
        symbol: The trading instrument symbol, e.g. ``'EURUSD'``, ``'XAUUSD'``.

    Returns:
        The contract size as a ``float``.

    Examples:
        >>> get_contract_size('EURUSD')
        100000.0
        >>> get_contract_size('XAUUSD')
        100.0
        >>> get_contract_size('XAGUSD')
        5000.0
        >>> get_contract_size('BTCUSD')
        1.0
    """
    sym = symbol.upper()

    if any(sym.startswith(prefix) for prefix in _CRYPTO_PREFIXES):
        return 1.0

    for metal, size in _METAL_CONTRACT.items():
        if metal in sym:
            return size

    return 100_000.0
