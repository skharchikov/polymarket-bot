"""Tests for the honest backtest harness modules."""

from backtest.fees import net_pnl
from backtest.fills import fill_price, max_fillable


def test_net_pnl_win_matches_live_formula():
    # stake 100 @ price 0.5, win: payout=200, exit fee=4, pnl=200-4-100-2=94
    assert abs(net_pnl(100.0, 0.5, True) - 94.0) < 1e-6


def test_net_pnl_loss_is_stake_plus_entry_fee():
    assert abs(net_pnl(100.0, 0.5, False) - (-102.0)) < 1e-6


def test_net_pnl_degenerate_prices_return_zero():
    assert net_pnl(100.0, 0.0, True) == 0.0
    assert net_pnl(100.0, 1.0, True) == 0.0
    assert net_pnl(0.0, 0.5, True) == 0.0


def test_fill_price_worse_than_quoted():
    p = fill_price(0.50, stake=100.0, liquidity_usd=10_000.0)
    assert 0.50 < p < 0.60


def test_fill_price_floor_applies_on_tiny_size():
    # even a negligible order pays at least the measured live floor
    p = fill_price(0.50, stake=1.0, liquidity_usd=1_000_000.0)
    assert p >= 0.50 + 0.0072 - 1e-9
    assert p <= 0.50 + 0.0072 + 1e-4  # extra impact is negligible


def test_fill_price_no_liquidity_uses_floor():
    assert abs(fill_price(0.30, stake=100.0, liquidity_usd=0.0) - (0.30 + 0.0072)) < 1e-9


def test_fill_price_clamped_below_one():
    assert fill_price(0.985, stake=100.0, liquidity_usd=1.0) <= 0.99


def test_max_fillable_caps_at_participation():
    assert abs(max_fillable(10_000.0, participation=0.10) - 1_000.0) < 1e-6
