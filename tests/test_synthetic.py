"""Algebraic checks on a hand-built panel, where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

from contfut.continuous import continuous_price
from contfut.data import RootPanel
from contfut.roll import selector


def make_panel(n=40, roll_at=20, gap=-0.5, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    front = pd.Series(100 + np.cumsum(rng.normal(0, .3, n)), index=idx)
    nxt = front + gap                       # a constant calendar spread
    px = pd.DataFrame({1: front, 2: nxt})
    tick = pd.DataFrame({1: ["AAH4"] * n, 2: ["AAM4"] * n}, index=idx)
    oi1 = pd.Series(np.r_[np.full(roll_at, 1e6), np.full(n - roll_at, 1e3)], index=idx)
    oi2 = pd.Series(np.r_[np.full(roll_at, 1e3), np.full(n - roll_at, 1e6)], index=idx)
    books = pd.DataFrame({1: oi1, 2: oi2})
    panel = RootPanel(root="AA", px=px, volume=books, oi=books, ticker=tick,
                      agg_oi=oi1 + oi2, calendar=idx, oi_raw=books, volume_raw=books,
                      stale_oi=pd.DataFrame(False, index=idx, columns=[1, 2]))
    rolls = pd.DataFrame([{
        "cycle": 0, "front_ticker": "AAH4", "next_ticker": "AAM4",
        "window_start": idx[0], "window_end": idx[-1], "provider_switch": pd.NaT,
        "roll_start": idx[roll_at - 1], "roll_date": idx[roll_at], "roll_end": idx[roll_at],
        "complete": True, "censored": False,
    }])
    return panel, rolls, idx, front, nxt


def test_gap_is_recovered_exactly():
    panel, rolls, idx, front, nxt = make_panel(gap=-0.5)
    sel = selector(panel, rolls)
    cp, gaps = continuous_price(panel, rolls, sel)
    assert gaps["gap"].iloc[0] == pytest.approx(-0.5)


def test_adjusted_series_has_no_jump_where_the_raw_one_does():
    panel, rolls, idx, front, nxt = make_panel(gap=-0.5, roll_at=20)
    sel = selector(panel, rolls)
    cp, _ = continuous_price(panel, rolls, sel)
    d = idx[20]
    # the held (unadjusted) series steps by the gap on the roll date ...
    assert cp["px_held"].diff().loc[d] == pytest.approx(front.diff().loc[d] - 0.5)
    # ... the back-adjusted one reproduces the front contract's own move
    assert cp["diff_adj_diff"].loc[d] == pytest.approx(front.diff().loc[d])
    assert cp["ret_splice"].loc[d] == pytest.approx(np.log(front.loc[d] / front.iloc[19]))


def test_back_adjustment_shifts_only_the_past():
    panel, rolls, idx, *_ = make_panel(gap=-0.5, roll_at=20)
    sel = selector(panel, rolls)
    cp, _ = continuous_price(panel, rolls, sel)
    after = cp["px_adj_diff"].iloc[20:] - cp["px_held"].iloc[20:]
    before = cp["px_adj_diff"].iloc[:20] - cp["px_held"].iloc[:20]
    assert np.allclose(after, 0.0)
    assert np.allclose(before, -0.5)


@pytest.mark.parametrize("gap", [-1.5, -0.25, 0.0, 0.75])
def test_invariants_hold_for_any_gap(gap):
    panel, rolls, idx, *_ = make_panel(gap=gap)
    sel = selector(panel, rolls)
    cp, _ = continuous_price(panel, rolls, sel)
    assert (cp["diff_adj_diff"] - cp["diff_splice"]).abs().max() < 1e-10
    assert (cp["ret_adj_ratio"] - cp["ret_splice"]).abs().max() < 1e-12
