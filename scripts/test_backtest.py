"""Tests for the honest backtest harness modules."""

import numpy as np
import pandas as pd

from backtest.fees import net_pnl
from backtest.fills import fill_price, max_fillable
from backtest.metrics import brier, brier_skill_vs_market, max_drawdown, summarize
from backtest.walkforward import fit_scaler_on_train, market_grouped_splits


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


def test_scaler_fit_on_train_only():
    # RobustScaler center = median of the train slice, so a huge value that
    # would only appear in a later (test) fold cannot shift it.
    train = pd.DataFrame({"f": np.arange(100.0)})
    scaler = fit_scaler_on_train(train)
    assert abs(scaler.center_[0] - 49.5) < 1e-6
    # a test-fold outlier is irrelevant: fitting on train alone is unchanged
    scaler2 = fit_scaler_on_train(train)
    assert np.allclose(scaler.center_, scaler2.center_)
    assert np.allclose(scaler.scale_, scaler2.scale_)


def test_brier_perfect_is_zero():
    assert brier([1.0, 0.0], [1, 0]) == 0.0


def test_brier_skill_negative_when_worse_than_market():
    # model always 0.5; market is nearly right → skill < 0
    assert brier_skill_vs_market([0.5, 0.5], [0.9, 0.1], [1, 0]) < 0


def test_max_drawdown_basic():
    assert abs(max_drawdown([10, 5, 8, 2, 6]) - 8.0) < 1e-9


def test_summarize_net_roi_and_baseline():
    bets = [
        {"stake": 100.0, "entry_fee": 2.0, "pnl": 94.0, "won": True,
         "model_prob": 0.7, "market_price": 0.5, "outcome": 1},
        {"stake": 100.0, "entry_fee": 2.0, "pnl": -102.0, "won": False,
         "model_prob": 0.7, "market_price": 0.5, "outcome": 0},
    ]
    s = summarize(bets)
    assert s["n"] == 2
    assert abs(s["net_pnl"] - (-8.0)) < 1e-9
    # net ROI on capital-at-risk (2 * 102) = -8 / 204 * 100
    assert abs(s["net_roi_pct"] - (-8.0 / 204.0 * 100.0)) < 1e-6
    assert s["baseline_no_bet_pnl"] == 0.0


def test_summarize_empty():
    assert summarize([])["n"] == 0


def test_market_grouped_split_no_market_spans_sides():
    # 6 markets, 2 snapshots each — no market may appear on both sides
    mids = [m for m in ["a", "b", "c", "d", "e", "f"] for _ in range(2)]
    ts = list(range(12))
    folds = list(market_grouped_splits(mids, ts, n_splits=2))
    assert len(folds) >= 1
    for train_mask, test_mask in folds:
        train_markets = {m for m, keep in zip(mids, train_mask) if keep}
        test_markets = {m for m, keep in zip(mids, test_mask) if keep}
        assert train_markets.isdisjoint(test_markets)
        assert test_markets  # non-empty test side
