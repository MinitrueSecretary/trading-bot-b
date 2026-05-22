"""Pip size utilities for various instrument types."""

# Instrument-type prefix lookup (checked in order; first match wins).
# Add new entries here to extend support without touching strategy code.
_CRYPTO_PREFIXES: tuple[str, ...] = ('BTC', 'ETH', 'LTC', 'XRP', 'BNB', 'SOL', 'ADA', 'DOGE')
_METAL_PIP: dict[str, float] = {
    'XAU': 0.01,   # Gold  — quoted to 2 d.p. (~$0.01 moves)
    'XAG': 0.001,  # Silver — quoted to 3 d.p. (~$0.001 moves)
    'XPT': 0.01,   # Platinum
    'XPD': 0.01,   # Palladium
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

    return 0.0001
