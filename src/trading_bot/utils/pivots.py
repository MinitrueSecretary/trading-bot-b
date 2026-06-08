"""Swing-point pivot detection utilities shared across strategies and analysers."""

import numpy as np
import numpy.typing as npt


def swing_lows(series: npt.NDArray[np.float64], window: int) -> list[int]:
    """Return indices of confirmed swing lows in *series*.

    A bar at index *i* is a swing low when its value is strictly less than
    every bar within ``window`` positions to the left **and** all available
    bars to the right (up to ``window``).  Bars near the right edge use
    however many bars exist rather than requiring a full ``window``, so
    pivots can be detected all the way up to the final bar.

    The last bar (*n-1*) is excluded because it may be an unfinished candle
    and cannot yet be confirmed as a local minimum.

    Args:
        series: 1-D array of values to scan.
        window: Number of bars required to the left; right side uses
            ``min(window, available)`` bars.

    Returns:
        Sorted list of integer indices that are confirmed swing lows.
    """
    indices: list[int] = []
    n = len(series)
    for i in range(window, n - 1):
        val = series[i]
        right = min(window, n - 1 - i)
        if all(val < series[i - j] for j in range(1, window + 1)) and all(
            val < series[i + j] for j in range(1, right + 1)
        ):
            indices.append(i)
    return indices


def swing_highs(series: npt.NDArray[np.float64], window: int) -> list[int]:
    """Return indices of confirmed swing highs in *series*.

    A bar at index *i* is a swing high when its value is strictly greater
    than every bar within ``window`` positions to the left **and** all
    available bars to the right (up to ``window``).  The last bar (*n-1*)
    is excluded as it may be an unfinished candle.

    Args:
        series: 1-D array of values to scan.
        window: Number of bars required to the left; right side uses
            ``min(window, available)`` bars.

    Returns:
        Sorted list of integer indices that are confirmed swing highs.
    """
    indices: list[int] = []
    n = len(series)
    for i in range(window, n - 1):
        val = series[i]
        right = min(window, n - 1 - i)
        if all(val > series[i - j] for j in range(1, window + 1)) and all(
            val > series[i + j] for j in range(1, right + 1)
        ):
            indices.append(i)
    return indices
