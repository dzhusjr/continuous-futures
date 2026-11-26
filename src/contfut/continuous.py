"""Continuous price and continuous open interest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RootPanel


def _by_ticker(panel: RootPanel, frame: pd.DataFrame) -> pd.Series:
    """Stack a per-rank frame into a (date, contract ticker) -> value lookup."""
    long = pd.concat([pd.DataFrame({"DATE": frame.index, "ticker": panel.ticker[g].values,
                                    "v": frame[g].values}) for g in frame.columns],
                     ignore_index=True).dropna(subset=["ticker"])
    return long.drop_duplicates(["DATE", "ticker"]).set_index(["DATE", "ticker"])["v"]


def _lookup(table: pd.Series, dates, tickers) -> pd.Series:
    idx = pd.MultiIndex.from_arrays([dates, pd.Series(tickers).values])
    return pd.Series(table.reindex(idx).values, index=dates)


def continuous_price(panel: RootPanel, rolls: pd.DataFrame, sel: pd.DataFrame):
    """Returns (series frame, roll gaps)."""
    idx, held = panel.px.index, sel["held_ticker"]
    px = _by_ticker(panel, panel.px)
    raw = _lookup(px, idx, held)

    # true P&L: price of yesterday's held contract, today and yesterday
    prev = held.shift(1).fillna(held.iloc[0])
    today_prev_contract = _lookup(px, idx, prev)
    splice_ret = np.log((today_prev_contract / raw.shift(1)).where(lambda r: r > 0))
    splice_diff = today_prev_contract - raw.shift(1)

    gaps = pd.DataFrame([
        {"roll_date": r["roll_date"],
         "p_front": px.get((r["roll_date"], r["front_ticker"])),
         "p_next": px.get((r["roll_date"], r["next_ticker"]))}
        for r in rolls.to_dict("records") if pd.notna(r["roll_date"])
    ]).dropna()
    if len(gaps):
        gaps = gaps.set_index("roll_date").assign(gap=lambda d: d.p_next - d.p_front,
                                                  ratio=lambda d: d.p_next / d.p_front)
    else:
        gaps = pd.DataFrame(columns=["p_front", "p_next", "gap", "ratio"])

    add, mul = pd.Series(0.0, index=idx), pd.Series(1.0, index=idx)
    for tr, row in gaps.iterrows():
        pos = idx.searchsorted(tr)
        add.iloc[:pos] += row["gap"]
        mul.iloc[:pos] *= row["ratio"]

    out = pd.DataFrame({"held_ticker": held, "px_held": raw,
                        "px_naive_gen1": pd.Series(panel.px[1].values, index=idx),
                        "px_adj_diff": raw + add, "px_adj_ratio": raw * mul,
                        "ret_splice": splice_ret, "diff_splice": splice_diff})
    for name, col in [("naive_gen1", "px_naive_gen1"), ("adj_diff", "px_adj_diff"),
                      ("adj_ratio", "px_adj_ratio")]:
        out[f"ret_{name}"] = np.log(out[col].where(out[col] > 0)).diff()
    out["diff_adj_diff"] = out["px_adj_diff"].diff()
    out["diff_naive_gen1"] = out["px_naive_gen1"].diff()
    return out, gaps


def continuous_oi(panel: RootPanel, sel: pd.DataFrame) -> pd.DataFrame:
    """The four positioning candidates, plus the front share diagnostic."""
    idx = panel.oi.index
    oi1, oi2, agg = panel.oi[1], panel.oi[2], panel.agg_oi.reindex(idx)
    # where generic 2 is stale, fall back on the identity OI_next = AGG - OI_front
    pair = (oi1.fillna(0) + oi2.fillna(0)).where(oi1.notna() | oi2.notna())
    pair = pair.mask(oi2.isna() & oi1.notna(), oi1 + (agg - oi1).clip(lower=0))

    out = pd.DataFrame({
        "oi_front_naive": panel.oi_raw[1],
        "oi_held": _lookup(_by_ticker(panel, panel.oi), idx, sel["held_ticker"]),
        "oi_active_pair": pair,
        "oi_aggregate": agg,
    })
    out["oi_continuous"] = out["oi_aggregate"].where(out["oi_aggregate"] > 0, out["oi_active_pair"])
    out["front_share_curve"] = (oi1 / agg).clip(0, 1)
    for c in ["oi_front_naive", "oi_held", "oi_active_pair", "oi_continuous"]:
        out[f"dlog_{c}"] = np.log(out[c].replace(0, np.nan)).diff()
    return out


def conviction(oi: pd.Series, window: int = 252, phase: pd.Series | None = None,
               deseasonalise: bool = False) -> pd.DataFrame:
    """Positioning signal: trailing z-score of log open interest."""
    x = np.log(oi.replace(0, np.nan))
    if deseasonalise and phase is not None:
        resid = x - x.rolling(window, min_periods=window // 4).mean()
        x = x - resid.groupby(phase).transform("mean").fillna(0.0)
    mu = x.rolling(window, min_periods=window // 4).mean()
    sd = x.rolling(window, min_periods=window // 4).std()
    return pd.DataFrame({"log_oi": x, "conviction_z": (x - mu) / sd})
