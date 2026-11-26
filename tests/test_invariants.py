"""The properties the construction must have, stated as assertions."""
import numpy as np
import pandas as pd


def test_roll_dates_ordered_within_cycle(built):
    r = built["rolls"]
    ok = r.dropna(subset=["roll_start", "roll_date", "roll_end"])
    assert (ok["roll_start"] <= ok["roll_date"]).all()
    assert (ok["roll_date"] <= ok["roll_end"]).all()


def test_roll_dates_strictly_increasing(built):
    d = built["rolls"]["roll_date"].dropna()
    assert d.is_monotonic_increasing and d.is_unique


def test_roll_date_inside_its_cycle(built):
    for _, r in built["rolls"].dropna(subset=["roll_date"]).iterrows():
        assert r["window_start"] <= r["roll_date"] <= r["window_end"]


def test_every_complete_cycle_gets_a_roll(built):
    c = built["rolls"]
    assert c[c["complete"]]["roll_date"].notna().all()


def test_roll_precedes_provider_switch(built):
    """Our roll is never later than the provider's: the liquidity moves first."""
    c = built["rolls"].dropna(subset=["roll_date", "provider_switch"])
    assert (c["roll_date"] <= c["provider_switch"]).all()


def test_each_contract_held_once_and_in_order(built):
    held = built["sel"]["held_ticker"].dropna()
    runs = held[held != held.shift()]
    assert runs.is_unique, "a contract is held, dropped and held again"


def test_selector_has_no_gaps(built):
    assert built["sel"]["held_ticker"].notna().all()
    assert built["sel"]["held_rank"].notna().all()


def test_held_contract_is_the_one_with_more_open_interest_at_the_roll(built):
    """On the roll date the contract we switch into must be the bigger book."""
    panel, rolls = built["panel"], built["rolls"]
    for _, r in rolls.dropna(subset=["roll_date"]).iterrows():
        d = r["roll_date"]
        oi1, oi2 = panel.oi.loc[d, 1], panel.oi.loc[d, 2]
        if np.isnan(oi1) or np.isnan(oi2):
            continue
        assert oi2 >= oi1, f"{built['root']} {d:%Y-%m-%d}: rolled into the smaller book"


def test_ratio_adjustment_preserves_returns_exactly(built):
    cp = built["cp"]
    err = (cp["ret_adj_ratio"] - cp["ret_splice"]).abs().max()
    assert err < 1e-12, f"ratio back-adjustment changed returns by {err}"


def test_difference_adjustment_preserves_price_changes_exactly(built):
    cp = built["cp"]
    err = (cp["diff_adj_diff"] - cp["diff_splice"]).abs().max()
    assert err < 1e-9, f"difference back-adjustment changed price moves by {err}"


def test_difference_adjustment_does_distort_returns(built):
    """The documented trade-off, asserted so it cannot be quietly forgotten."""
    cp = built["cp"]
    err = (cp["ret_adj_diff"] - cp["ret_splice"]).abs().max()
    assert err > 0, "difference adjustment unexpectedly preserved returns"


def test_unadjusted_provider_series_injects_a_jump(built):
    cp, flags = built["cp"], built["flags"]
    e = (cp["ret_naive_gen1"] - cp["ret_splice"]).abs()
    on_switch = e[flags["is_provider_switch"]].max()
    assert on_switch > 10e-4, "expected a visible jump in the unadjusted series"


def test_continuous_price_is_complete_and_positive(built):
    cp = built["cp"]
    assert cp["px_adj_diff"].notna().all()
    assert (cp["px_adj_ratio"] > 0).all()


def test_roll_gap_measured_same_day(built):
    """The adjustment must come from two prices observed on the SAME date."""
    panel, gaps = built["panel"], built["gaps"]
    for d, row in gaps.iterrows():
        assert d in panel.px.index
        assert np.isfinite(row["p_front"]) and np.isfinite(row["p_next"])


def test_aggregate_equals_front_pair_for_bonds(built):
    """Quarterly bond OI lives in the front two contracts; this is what licenses using the curve aggregate as the positioning series."""
    oi = built["oi"]
    live = oi["oi_active_pair"].notna() & (oi["oi_aggregate"] > 0)
    ratio = (oi.loc[live, "oi_active_pair"] / oi.loc[live, "oi_aggregate"])
    assert abs(ratio.median() - 1.0) < 0.01


def test_chosen_oi_has_no_roll_cliff(built):
    """The chosen positioning series must not move more than 25% in a day at a roll, whereas the naive one moves by an order of magnitude."""
    oi, flags = built["oi"], built["flags"]
    roll = flags["is_roll_date"]
    chosen = oi.loc[roll, "dlog_oi_continuous"].abs().max()
    naive = oi.loc[roll, "dlog_oi_front_naive"].abs().max()
    assert chosen < 0.25, f"chosen OI jumped {chosen:.2f} in log terms at a roll"
    assert naive > 3 * chosen


def test_front_share_bounded(built):
    s = built["oi"]["front_share_curve"].dropna()
    assert ((s >= 0) & (s <= 1)).all()


def test_no_forward_filled_days_survive(built):
    """Every retained date must show activity somewhere on the curve."""
    panel = built["panel"]
    act = pd.concat([panel.px, panel.volume_raw, panel.oi_raw], axis=1)
    changed = (act != act.shift()).any(axis=1)
    assert changed.iloc[1:].all()


def test_weekends_removed(built):
    assert (built["panel"].px.index.dayofweek < 5).all()


def test_stale_filter_leaves_front_month_alone(built):
    """The staleness filter must not be eating live front-month data."""
    panel = built["panel"]
    frac = panel.stale_oi[1].mean()
    assert frac < 0.20, f"{frac:.1%} of front-month OI flagged stale"
