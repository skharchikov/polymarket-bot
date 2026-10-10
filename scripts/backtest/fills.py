"""Conservative fill / slippage model.

The Data-API price history we backtest on has no L2 depth, so this is a
deliberately conservative proxy anchored to the slippage we actually measured
live (baseline doc: p95 = 0.0072). Price impact grows with the stake's share of
available liquidity; a bet is also capped at a fraction of that liquidity.
"""


def max_fillable(liquidity_usd, participation=0.10):
    """Cap a bet at a fraction of available liquidity."""
    return max(0.0, liquidity_usd * participation)


def fill_price(quoted, stake, liquidity_usd, slippage_floor=0.0072):
    """Effective price paid, always >= quoted, clamped below 0.99."""
    if liquidity_usd <= 0.0:
        impact = slippage_floor
    else:
        impact = slippage_floor + 0.05 * (stake / liquidity_usd)
    return min(0.99, quoted + impact)
