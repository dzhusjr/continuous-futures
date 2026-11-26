#!/usr/bin/env python3
"""End-to-end run: raw CSV -> every table and headline number."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contfut.config import BOND_SPECS, DEFAULT_ROLL_PARAMS, SPECS, RollParams
from contfut.continuous import continuous_oi, continuous_price, conviction
from contfut.data import build_panel, data_quality_report, load_raw
from contfut.metrics import oi_metric_table, price_metric_table, signal_metric_table
from contfut.roll import detect_rolls, roll_flags, selector

ROLL_COLS = ["cycle", "front_ticker", "next_ticker", "roll_start", "roll_date", "roll_end",
             "roll_len_days", "vol_cross", "cal_anchor", "cal_anchor_kind", "cal_roll",
             "provider_switch", "gap_oi_vs_provider_bd", "gap_oi_vs_cal_bd",
             "gap_oi_vs_vol_bd", "complete", "censored"]
PROVIDER, DIFFADJ = "provider generic-1 splice", "difference (Panama) back-adjusted"
NAIVE_OI, CHOSEN_OI = "generic-1 open interest", "curve aggregate OI (chosen)"
DESEASON = "curve aggregate, roll-phase deseasonalised"


def analyse(df, root, n_perm=10_000, params=DEFAULT_ROLL_PARAMS):
    """The whole pipeline for one root."""
    panel = build_panel(df, root)
    rolls = detect_rolls(panel, params)
    sel, flags = selector(panel, rolls), roll_flags(panel, rolls)
    cp, gaps = continuous_price(panel, rolls, sel)
    oi = continuous_oi(panel, sel)
    return dict(root=root, panel=panel, rolls=rolls, sel=sel, flags=flags, cp=cp, gaps=gaps, oi=oi,
                price_metrics=price_metric_table(cp, flags, n_perm),
                oi_metrics=oi_metric_table(oi, flags, n_perm),
                signal_metrics=signal_metric_table(oi, flags))


def summarise(a) -> dict:
    """Headline numbers, pulled straight out of the metric tables."""
    rc = a["rolls"][a["rolls"]["complete"]]
    pm = a["price_metrics"].set_index("series")
    om = a["oi_metrics"].set_index("series")
    sm = a["signal_metrics"].set_index("signal")
    gap_bp = (a["gaps"]["gap"] / a["gaps"]["p_front"]).abs() * 1e4
    return {
        "root": a["root"], "name": SPECS[a["root"]].name,
        "exchange": SPECS[a["root"]].exchange, "anchor": SPECS[a["root"]].anchor,
        "cycles": len(rc),
        "roll_len_days_median": rc["roll_len_days"].median(),
        "gap_vs_provider_bd_median": rc["gap_oi_vs_provider_bd"].median(),
        "gap_vs_calendar_bd_median": rc["gap_oi_vs_cal_bd"].median(),
        "gap_vs_volume_bd_median": rc["gap_oi_vs_vol_bd"].median(),
        "pct_days_generic1_wrong": 100 * (a["sel"]["held_rank"] == 2).mean(),
        "roll_gap_bp_mean": gap_bp.mean(), "roll_gap_bp_max": gap_bp.max(),
        "cum_backadjust_points": a["gaps"]["gap"].sum(),
        "RAS_price_provider": pm.loc[PROVIDER, "ras"],
        "RAS_price_provider_p": pm.loc[PROVIDER, "p_value"],
        "injected_jump_bp_provider": pm.loc[PROVIDER, "rms_bp"],
        "injected_jump_sigma_provider": pm.loc[PROVIDER, "rms_sigma"],
        "RAS_price_diffadj": pm.loc[DIFFADJ, "ras"],
        "RAS_price_diffadj_p": pm.loc[DIFFADJ, "p_value"],
        "injected_jump_bp_diffadj": pm.loc[DIFFADJ, "rms_bp"],
        "RAS_oi_generic1": om.loc[NAIVE_OI, "ras"], "RAS_oi_chosen": om.loc[CHOSEN_OI, "ras"],
        "calendar_r2_generic1": sm.loc[NAIVE_OI, "calendar_r2"],
        "calendar_r2_chosen": sm.loc[CHOSEN_OI, "calendar_r2"],
        "calendar_r2_deseason": sm.loc[DESEASON, "calendar_r2"],
        "corr_generic1_vs_chosen": sm.loc[NAIVE_OI, "corr_with_chosen"],
    }


def stir_evidence(df, stirs=("SFR", "ER", "SFI"), bonds=("TY", "RX")):
    """Part 2: how open interest is spread across the strip."""
    rows = []
    for root in list(stirs) + list(bonds):
        panel = build_panel(df, root)
        ranks = sorted(panel.oi_raw.columns)
        cum = panel.oi_raw[ranks].div(panel.agg_oi, axis=0).cumsum(axis=1)
        deep = [r for r in ranks if r >= 9]
        rows.append({
            "root": root, "name": SPECS[root].name, "max_rank": max(ranks),
            "front_share_median": cum[1].median(),
            **{f"front{k}_share_median": cum[k].median() if k in cum else np.nan
               for k in (2, 4, 8)},
            "front16_share_median": cum[max(ranks)].median() if max(ranks) > 8 else np.nan,
            "rank_with_max_oi_mode": int(panel.oi_raw[ranks].idxmax(axis=1).mode().iloc[0]),
            "oi_nan_rank_ge_9_pct": 100 * panel.oi_raw[deep].isna().mean().mean() if deep else np.nan,
        })
    return pd.DataFrame(rows)


def commodity_evidence(df, roots=("CL", "CO", "QS", "GC")):
    """Part 2: roll-gap magnitudes on commodities."""
    relaxed = RollParams(start_threshold=0.60)
    rows = []
    for root in roots:
        panel = build_panel(df, root)
        default_rolls, rr = detect_rolls(panel, DEFAULT_ROLL_PARAMS), detect_rolls(panel, relaxed)
        _, gaps = continuous_price(panel, rr, selector(panel, rr))
        gap_bp = (gaps["gap"] / gaps["p_front"]) * 1e4
        share = panel.oi[1] / (panel.oi[1] + panel.oi[2])
        years = (panel.px.index[-1] - panel.px.index[0]).days / 365.25
        rows.append({
            "root": root, "name": SPECS[root].name, "cycle": SPECS[root].cycle,
            "cycles": len(default_rolls),
            "rolls_detected_default_thresholds": int(default_rolls["roll_date"].notna().sum()),
            "rolls_detected_relaxed_0p60": int(rr["roll_date"].notna().sum()),
            "front_pair_share_max": share.max(),
            "front_share_of_curve_median": (panel.oi[1] / panel.agg_oi).median(),
            "rolls_per_year": round(int(rr["roll_date"].notna().sum()) / years, 1),
            "roll_gap_bp_mean": gap_bp.mean(), "roll_gap_bp_abs_mean": gap_bp.abs().mean(),
            "roll_gap_bp_max_abs": gap_bp.abs().max(),
            "cum_backadjust_points": gaps["gap"].sum(),
            "last_price": panel.px[1].iloc[-1],
            "cum_adj_pct_of_last_price": 100 * gaps["gap"].sum() / panel.px[1].iloc[-1],
            "min_price_in_sample": panel.px.min().min(),
        })
    return pd.DataFrame(rows)


def run_root(df, root, out, n_perm):
    """Analyse one root, write its tables, return its summary."""
    print(f"analysing {root} ({SPECS[root].name})")
    a = analyse(df, root, n_perm)
    a["rolls"][ROLL_COLS].to_csv(out / "tables" / f"roll_schedule_{root}.csv", index=False)
    for key in ["price_metrics", "oi_metrics", "signal_metrics"]:
        a[key].to_csv(out / "tables" / f"{key.replace('_metrics', '')}_metrics_{root}.csv", index=False)
    pd.DataFrame({
        "px_adj_diff": a["cp"]["px_adj_diff"], "px_adj_ratio": a["cp"]["px_adj_ratio"],
        "ret_splice": a["cp"]["ret_splice"], "held_ticker": a["cp"]["held_ticker"],
        "oi_continuous": a["oi"]["oi_continuous"],
        "conviction_z": conviction(a["oi"]["oi_continuous"])["conviction_z"],
        "in_roll": a["flags"]["in_roll"],
    }).to_csv(out / "tables" / f"continuous_series_{root}.csv")
    return summarise(a)


def sweep_bonds(df, out, n_perm=2000):
    """Part 1.6: every bond root through the same definition, nothing retuned."""
    print("sweeping all bond roots ...")
    rows = []
    for root in BOND_SPECS:
        a = analyse(df, root, n_perm)
        rows.append(summarise(a))
    sw = pd.DataFrame(rows)
    sw.to_csv(out / "tables" / "cross_root_summary.csv", index=False)
    return {"n_roots": len(sw), "roll_len_days_median": sw["roll_len_days_median"].median(),
            "gap_vs_calendar_bd_max_abs": sw["gap_vs_calendar_bd_median"].abs().max(),
            "pct_days_generic1_wrong_min": sw["pct_days_generic1_wrong"].min(),
            "pct_days_generic1_wrong_max": sw["pct_days_generic1_wrong"].max()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/FUT_raw_data.csv")
    ap.add_argument("--primary", default="TY")
    ap.add_argument("--secondary", default="RX")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--quick", action="store_true", help="fewer permutations, skip the sweep")
    args = ap.parse_args()
    n_perm = 1000 if args.quick else args.n_perm

    out = Path(args.outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    print(f"loading {args.data} ...")
    df = load_raw(args.data)
    data_quality_report(df).to_csv(out / "tables" / "data_quality.csv", index=False)

    head = {root: run_root(df, root, out, n_perm) for root in (args.primary, args.secondary)}
    if not args.quick:
        head["cross_root"] = sweep_bonds(df, out)

    print("Part 2 evidence ...")
    stir, com = stir_evidence(df), commodity_evidence(df)
    stir.to_csv(out / "tables" / "part2_stir_oi_distribution.csv", index=False)
    com.to_csv(out / "tables" / "part2_commodity_roll_gaps.csv", index=False)
    head["part2"] = {"stir": stir.to_dict("records"), "commodity": com.to_dict("records")}

    (out / "headline_numbers.json").write_text(json.dumps(head, indent=2, default=str))
    print(f"\nwrote tables to {out}/")
    print(json.dumps({k: v for k, v in head.items() if k != "part2"}, indent=2, default=str)[:1200])


if __name__ == "__main__":
    main()
