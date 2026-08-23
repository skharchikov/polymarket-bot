"""Tests for the honest backtest harness modules."""

from backtest.fees import net_pnl


def test_net_pnl_win_matches_live_formula():
    # stake 100 @ price 0.5, win: payout=200, exit fee=4, pnl=200-4-100-2=94
    assert abs(net_pnl(100.0, 0.5, True) - 94.0) < 1e-6


def test_net_pnl_loss_is_stake_plus_entry_fee():
    assert abs(net_pnl(100.0, 0.5, False) - (-102.0)) < 1e-6


def test_net_pnl_degenerate_prices_return_zero():
    assert net_pnl(100.0, 0.0, True) == 0.0
    assert net_pnl(100.0, 1.0, True) == 0.0
    assert net_pnl(0.0, 0.5, True) == 0.0
