"""Roll = first session the OI front share s = OI1/(OI1+OI2) stays below 0.5,
after having been above 0.85 in the same cycle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendars import roll_anchor_date, shift_bd
from .config import CALENDAR_OFFSET_BD, DEFAULT_ROLL_PARAMS, RollParams
from .data import RootPanel


def first_sustained(series: pd.Series, threshold: float, confirm: int = 2) -> pd.Timestamp | None:
    """First index where `series` goes below `threshold` and stays below it for `confirm` observations."""
    s = series.dropna()
    below = (s < threshold).to_numpy()
    for i in range(len(below)):
        if below[i] and below[i:i + confirm].all():
            return s.index[i]
    return None


def cycle_windows(panel: RootPanel) -> pd.DataFrame:
    """One row per front-contract cycle, bounded by the provider's own switches."""
    tk = panel.ticker[1]
    starts = tk.index[(tk != tk.shift()).to_numpy() | (np.arange(len(tk)) == 0)]
    rows = []
    for i, d0 in enumerate(starts):
        d1 = starts[i + 1] if i + 1 < len(starts) else None
        seg = panel.px.index[(panel.px.index >= d0) & ((panel.px.index < d1) if d1 is not None else True)]
        if len(seg):
            rows.append({"cycle": i, "front_ticker": tk.loc[d0], "next_ticker": panel.ticker[2].loc[d0],
                         "window_start": d0, "window_end": seg[-1], "provider_switch": d1,
                         "n_days": len(seg)})
    return pd.DataFrame(rows)


def detect_rolls(panel: RootPanel, params: RollParams = DEFAULT_ROLL_PARAMS) -> pd.DataFrame:
    """OI roll per cycle, with volume and calendar rules as comparators."""
    pair = panel.oi[1] + panel.oi[2]
    s_oi = (panel.oi[1] / pair).where(pair >= params.min_pair_oi)
    v_pair = panel.volume[1] + panel.volume[2]
    s_vol = (panel.volume[1] / v_pair).where(v_pair > 0).rolling(3, min_periods=1).mean()
    cal = panel.calendar

    def bd(a, b):
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            return np.nan
        return int(cal.searchsorted(b) - cal.searchsorted(a))

    out = []
    for c in cycle_windows(panel).to_dict("records"):
        seg = s_oi.loc[c["window_start"]:c["window_end"]]
        above = seg[seg >= params.start_threshold]
        censored = above.empty
        dates = {"roll_start": None, "roll_date": None, "roll_end": None, "vol_cross": None}
        if not censored:
            seg = seg.loc[above.index[0]:]
            segv = s_vol.loc[above.index[0]:c["window_end"]]
            for key, thr in [("roll_start", params.start_threshold),
                             ("roll_date", params.roll_threshold),
                             ("roll_end", params.end_threshold)]:
                dates[key] = first_sustained(seg, thr, params.confirm_days)
            dates["vol_cross"] = first_sustained(segv, 0.5, params.confirm_days)

        anchor, kind = roll_anchor_date(panel.root, c["front_ticker"], ref_date=c["window_start"])
        cal_roll = shift_bd(cal, anchor, -CALENDAR_OFFSET_BD) if anchor is not None else None
        out.append({**c, **dates, "censored": bool(censored),
                    "complete": bool(c["cycle"] > 0 and pd.notna(c["provider_switch"]) and not censored),
                    "roll_len_days": bd(dates["roll_start"], dates["roll_end"]),
                    "cal_anchor": anchor, "cal_anchor_kind": kind, "cal_roll": cal_roll,
                    "gap_oi_vs_provider_bd": bd(dates["roll_date"], c["provider_switch"]),
                    "gap_oi_vs_cal_bd": bd(dates["roll_date"], cal_roll),
                    "gap_oi_vs_vol_bd": bd(dates["roll_date"], dates["vol_cross"]),
                    "root": panel.root})
    return pd.DataFrame(out)


def roll_flags(panel: RootPanel, rolls: pd.DataFrame) -> pd.DataFrame:
    """Per-session roll flags and phase relative to the roll date."""
    idx = panel.px.index
    f = pd.DataFrame({"in_roll": False, "is_roll_date": False, "is_provider_switch": False,
                      "roll_phase": np.nan}, index=idx)
    for r in rolls.to_dict("records"):
        if pd.notna(r["roll_start"]) and pd.notna(r["roll_end"]):
            f.loc[r["roll_start"]:r["roll_end"], "in_roll"] = True
        if pd.notna(r.get("provider_switch")) and r["provider_switch"] in idx:
            f.loc[r["provider_switch"], "is_provider_switch"] = True
        if pd.notna(r["roll_date"]):
            f.loc[r["roll_date"], "is_roll_date"] = True
            pos = idx.searchsorted(r["roll_date"])
            lo, hi = max(0, pos - 20), min(len(idx), pos + 21)
            f.iloc[lo:hi, f.columns.get_loc("roll_phase")] = np.arange(lo - pos, hi - pos)
    return f


def selector(panel: RootPanel, rolls: pd.DataFrame) -> pd.DataFrame:
    """Held contract and its generic rank on each session."""
    idx = panel.px.index
    held = pd.Series(index=idx, dtype=object)
    for r in rolls.to_dict("records"):
        seg = idx[(idx >= r["window_start"]) & (idx <= r["window_end"])]
        cut = r["roll_date"] if pd.notna(r["roll_date"]) else seg[-1] + pd.Timedelta(days=1)
        held.loc[seg[seg < cut]] = r["front_ticker"]
        held.loc[seg[seg >= cut]] = r["next_ticker"]
    rank = pd.Series(np.nan, index=idx)
    for g in panel.ticker.columns:
        rank = rank.mask(panel.ticker[g].eq(held) & rank.isna(), g)
    return pd.DataFrame({"held_ticker": held, "held_rank": rank})
