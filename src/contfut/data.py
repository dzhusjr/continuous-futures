"""Loading and cleaning."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ACTIVITY_COLS, SPECS

RAW_COLUMNS = ["DATE", "ID", "ROOT", "GEN_RANK", "PX_LAST", "PX_HIGH", "PX_LOW",
               "PX_VOLUME", "OPEN_INT", "FUT_AGGTE_OPEN_INT", "FUT_CUR_GEN_TICKER"]
NUMERIC = ["PX_LAST", "PX_HIGH", "PX_LOW", "PX_VOLUME", "OPEN_INT", "FUT_AGGTE_OPEN_INT"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if missing := set(RAW_COLUMNS) - set(df.columns):
        raise ValueError(f"missing expected columns: {sorted(missing)}")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df["FUT_CUR_GEN_TICKER"] = df["FUT_CUR_GEN_TICKER"].astype(str).str.strip()
    df["ROOT"] = df["ROOT"].astype(str).str.strip()
    df["GEN_RANK"] = df["GEN_RANK"].astype(int)
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    return df.sort_values(["ROOT", "GEN_RANK", "DATE"]).reset_index(drop=True)


def _moved(x: pd.DataFrame) -> pd.Series:
    """Row-wise: did anything change versus the previous row (NaN == NaN)?"""
    prev = x.shift()
    return (~((x == prev) | (x.isna() & prev.isna()))).any(axis=1)


def traded_calendar(root_df: pd.DataFrame, activity_cols=ACTIVITY_COLS) -> pd.DatetimeIndex:
    """Sessions on which any generic of the root changed."""
    d = root_df.sort_values(["GEN_RANK", "DATE"])
    changed = (d.groupby("GEN_RANK")[list(activity_cols)].apply(_moved)
               .reset_index(level=0, drop=True).sort_index())
    flag = d.assign(_c=changed).groupby("DATE")["_c"].any()
    flag.iloc[0] = True                               # no predecessor for the first date
    idx = pd.DatetimeIndex(flag[flag].index)
    return idx[idx.dayofweek < 5]


def stale_mask(values: pd.DataFrame, tickers: pd.DataFrame, max_run: int = 2) -> pd.DataFrame:
    """True where a value is carried forward rather than freshly printed."""
    out = {}
    for col in values.columns:
        v = values[col]
        run = (v != v.shift()).cumsum()
        long_run = (run.map(run.value_counts()) > max_run) & (v.groupby(run).cumcount() >= max_run)
        carried = pd.Series(False, index=v.index)
        if col in tickers.columns:
            new_contract = (tickers[col] != tickers[col].shift()) & (v == v.shift())
            # a carry stays invalid until the value finally moves
            carried = new_contract | (new_contract.where(new_contract).ffill().fillna(False)
                                      .astype(bool) & (v == v.shift()))
        out[col] = (long_run | carried).fillna(False)
    return pd.DataFrame(out, index=values.index)


@dataclass
class RootPanel:
    """Wide, sessions-only view of one root."""
    root: str
    px: pd.DataFrame
    volume: pd.DataFrame          # staleness-filtered
    oi: pd.DataFrame              # staleness-filtered
    ticker: pd.DataFrame
    agg_oi: pd.Series             # whole-curve OI
    calendar: pd.DatetimeIndex
    oi_raw: pd.DataFrame = None
    volume_raw: pd.DataFrame = None
    stale_oi: pd.DataFrame = None



def build_panel(df: pd.DataFrame, root: str, drop_ffilled: bool = True,
                drop_stale: bool = True, max_run: int = 2) -> RootPanel:
    r = df[df["ROOT"] == root]
    if r.empty:
        raise KeyError(f"root {root!r} not in data")
    cal = traded_calendar(r)
    if drop_ffilled:
        r = r[r["DATE"].isin(cal)]
    wide = lambda col: r.pivot(index="DATE", columns="GEN_RANK", values=col)
    oi_raw, vol_raw, tick = wide("OPEN_INT"), wide("PX_VOLUME"), wide("FUT_CUR_GEN_TICKER")
    stale = stale_mask(oi_raw, tick, max_run)
    return RootPanel(
        root=root, px=wide("PX_LAST"), ticker=tick, calendar=cal,
        oi=oi_raw.mask(stale) if drop_stale else oi_raw,
        volume=vol_raw.mask(stale_mask(vol_raw, tick, max_run)) if drop_stale else vol_raw,
        agg_oi=r[r["GEN_RANK"] == r["GEN_RANK"].min()].set_index("DATE")["FUT_AGGTE_OPEN_INT"].astype(float),
        oi_raw=oi_raw, volume_raw=vol_raw, stale_oi=stale)


def data_quality_report(df: pd.DataFrame, roots: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for root in roots or sorted(df["ROOT"].unique()):
        r = df[df["ROOT"] == root]
        p = build_panel(df, root)
        spec = SPECS.get(root)
        rows.append({
            "root": root, "name": spec.name if spec else "",
            "asset_class": spec.asset_class if spec else "",
            "start": r["DATE"].min().date(), "end": r["DATE"].max().date(),
            "max_gen_rank": int(r["GEN_RANK"].max()),
            "calendar_days": int(r["DATE"].nunique()), "trading_days": len(p.calendar),
            "ffilled_days": int(r["DATE"].nunique()) - len(p.calendar),
            "nan_oi_rank1_2": int(r[r["GEN_RANK"] <= 2]["OPEN_INT"].isna().sum()),
            "nan_oi_rank3plus": int(r[r["GEN_RANK"] >= 3]["OPEN_INT"].isna().sum()),
            "nonpositive_px": int((r["PX_LAST"] <= 0).sum()),
            "stale_oi_rank2_days": int(p.stale_oi[2].sum()) if 2 in p.stale_oi else 0,
        })
    return pd.DataFrame(rows)
