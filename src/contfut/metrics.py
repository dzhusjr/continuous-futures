"""Roll-robustness metrics: RAS, injected jump, calendar R^2."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SERIES = [("provider generic-1 splice", "ret_naive_gen1", "is_provider_switch"),
                ("difference (Panama) back-adjusted", "ret_adj_diff", "is_roll_date"),
                ("ratio back-adjusted", "ret_adj_ratio", "is_roll_date"),
                ("splice returns (benchmark)", "ret_splice", "is_roll_date")]

OI_SERIES = [("generic-1 open interest", "oi_front_naive"),
             ("held-contract open interest", "oi_held"),
             ("active pair OI(1)+OI(2)", "oi_active_pair"),
             ("curve aggregate OI (chosen)", "oi_continuous")]


def trailing_vol(x: pd.Series, exclude: pd.Series | None = None,
                 window: int = 60, min_periods: int = 20) -> pd.Series:
    """Lagged rolling std, ignoring `exclude` sessions."""
    y = x if exclude is None else x.mask(exclude.reindex(x.index).fillna(False))
    return y.rolling(window, min_periods=min_periods).std().shift(1)


def _z(x, exclude, window):
    return (x / trailing_vol(x, exclude, window)).replace([np.inf, -np.inf], np.nan)


def artefact_score(x: pd.Series, roll_days: pd.Series, exclude_from_vol=None,
                   window: int = 60, n_perm: int = 10_000, seed: int = 0) -> dict:
    roll_days = roll_days.reindex(x.index).fillna(False).astype(bool)
    z = _z(x, exclude_from_vol if exclude_from_vol is not None else roll_days, window)
    ok = z.notna()
    zr, zo = z[ok & roll_days].abs(), z[ok & ~roll_days].abs()
    if zr.empty or zo.empty:
        return {"ras": np.nan, "median_z_roll": np.nan, "median_z_other": np.nan,
                "p_value": np.nan, "n_roll": len(zr)}
    mr, mo = float(zr.median()), float(zo.median())
    stat = mr / mo - 1.0
    rng, pool = np.random.default_rng(seed), z[ok].abs().to_numpy()
    draws = np.array([np.median(rng.choice(pool, len(zr), replace=False)) / mo - 1.0
                      for _ in range(n_perm)])
    return {"ras": stat, "median_z_roll": mr, "median_z_other": mo,
            "p_value": float((np.abs(draws) >= abs(stat)).mean()), "n_roll": len(zr)}


def injected_jump(candidate: pd.Series, benchmark: pd.Series, roll_days: pd.Series) -> dict:
    e = (candidate - benchmark).replace([np.inf, -np.inf], np.nan)
    e = e[roll_days.reindex(e.index).fillna(False).astype(bool)].dropna()
    if e.empty:
        return dict.fromkeys(["rms_bp", "max_bp", "rms_sigma", "max_sigma"], np.nan)
    rms, mx, sd = float(np.sqrt((e ** 2).mean())), float(e.abs().max()), float(benchmark.std())
    return {"rms_bp": rms * 1e4, "max_bp": mx * 1e4, "rms_sigma": rms / sd, "max_sigma": mx / sd}


def calendar_r2(signal: pd.Series, phase: pd.Series) -> float:
    """R^2 of the signal on one dummy per session-relative-to-roll."""
    d = pd.DataFrame({"y": signal, "phase": phase.reindex(signal.index)}).dropna()
    if d.empty or d["phase"].nunique() < 2:
        return np.nan
    ss_tot = float(((d["y"] - d["y"].mean()) ** 2).sum())
    ss_res = float(((d["y"] - d.groupby("phase")["y"].transform("mean")) ** 2).sum())
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _split_mean(x: pd.Series, flag: pd.Series):
    flag, ok = flag.reindex(x.index).fillna(False).astype(bool), x.notna()
    inside = float(x[ok & flag].mean()) if (ok & flag).any() else np.nan
    outside = float(x[ok & ~flag].mean()) if (ok & ~flag).any() else np.nan
    return inside, outside


def signal_jump_enrichment(z: pd.Series, in_roll: pd.Series) -> dict:
    """Mean |daily change| of the signal inside vs outside roll windows."""
    m_in, m_out = _split_mean(z.diff().abs(), in_roll)
    return {"mean_abs_dz_in_roll": m_in, "mean_abs_dz_out": m_out,
            "dz_ratio": m_in / m_out if m_out else np.nan}


def conviction_enrichment(z: pd.Series, in_roll: pd.Series, k: float = 2.0) -> dict:
    p_in, p_out = _split_mean((z.abs() > k).where(z.notna()), in_roll)
    return {"p_event_in_roll": p_in, "p_event_out": p_out,
            "enrichment": p_in / p_out if p_out else np.nan}


def price_metric_table(cp: pd.DataFrame, flags: pd.DataFrame, n_perm: int = 10_000) -> pd.DataFrame:
    """Each price construction scored on its own roll days."""
    excl = flags["in_roll"] | flags["is_provider_switch"]
    return pd.DataFrame([
        {"series": name, "event_days": ev,
         **artefact_score(cp[col], flags[ev], exclude_from_vol=excl, n_perm=n_perm),
         **injected_jump(cp[col], cp["ret_splice"], flags[ev])}
        for name, col, ev in PRICE_SERIES])


def oi_metric_table(oi: pd.DataFrame, flags: pd.DataFrame, n_perm: int = 10_000) -> pd.DataFrame:
    rows = []
    for name, col in OI_SERIES:
        x, roll = oi[f"dlog_{col}"], flags["is_roll_date"]
        rows.append({"series": name, **artefact_score(x, roll, n_perm=n_perm),
                     "median_abs_dlog_roll": float(x[roll].abs().median()),
                     "median_abs_dlog_other": float(x[~roll].abs().median()),
                     "max_abs_dlog_roll": float(x[roll].abs().max())})
    return pd.DataFrame(rows)


def signal_metric_table(oi: pd.DataFrame, flags: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Conviction signal quality for each OI construction."""
    from .continuous import conviction
    specs = [(n, c, False) for n, c in OI_SERIES]
    specs.append(("curve aggregate, roll-phase deseasonalised", "oi_continuous", True))
    ref = conviction(oi["oi_continuous"], window)["conviction_z"]
    rows = []
    for name, col, des in specs:
        z = conviction(oi[col], window, phase=flags["roll_phase"],
                       deseasonalise=des)["conviction_z"]
        rows.append({"signal": name, "corr_with_chosen": float(z.corr(ref)),
                     "calendar_r2": calendar_r2(z, flags["roll_phase"]),
                     **signal_jump_enrichment(z, flags["in_roll"]),
                     **conviction_enrichment(z, flags["in_roll"]),
                     "signal_sd": float(z.std())})
    return pd.DataFrame(rows)
