"""Contract-code parsing and exchange delivery dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EUREX_BONDS, MONTH_CODES, SPECS, US_TREASURIES


def parse_contract(ticker: str, ref_date=None) -> tuple[str, int, int]:
    """'TYH20' -> ('TY', 2020, 3)."""
    t = str(ticker).strip()
    if len(t) >= 2 and t[-2].upper() in MONTH_CODES and t[-1].isdigit():
        code, root = t[-2].upper(), t[:-2].strip()
        ref = pd.Timestamp(ref_date).year if ref_date is not None else pd.Timestamp.today().year
        year = min((y for y in range(ref - 5, ref + 6) if y % 10 == int(t[-1])),
                   key=lambda y: abs(y - ref))
    elif len(t) >= 3 and t[-3].upper() in MONTH_CODES and t[-2:].isdigit():
        code, root, yy = t[-3].upper(), t[:-3].strip(), int(t[-2:])
        year = 2000 + yy if yy < 70 else 1900 + yy
    else:
        raise ValueError(f"unparseable contract ticker {ticker!r}")
    return root, year, MONTH_CODES[code]


def shift_bd(cal, anchor, n: int) -> pd.Timestamp | None:
    """Move `n` sessions from `anchor` along the traded calendar."""
    cal = pd.DatetimeIndex(cal)
    pos = cal.searchsorted(pd.Timestamp(anchor), side="right") - 1 + n
    return cal[pos] if 0 <= pos < len(cal) else None


def _last_bd(year: int, month: int) -> pd.Timestamp:
    eom = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    return pd.Timestamp(np.busday_offset(eom.date(), 0, roll="backward"))


def _prev_month(year: int, month: int) -> tuple[int, int]:
    d = pd.Timestamp(year=year, month=month, day=1) - pd.offsets.MonthBegin(1)
    return d.year, d.month


def first_notice_day(root: str, year: int, month: int) -> pd.Timestamp | None:
    if root in US_TREASURIES:
        return _last_bd(*_prev_month(year, month))
    if root == "G":
        first = pd.Timestamp(year=year, month=month, day=1)
        return pd.Timestamp(np.busday_offset(first.date(), -2, roll="backward"))
    if root in SPECS and SPECS[root].asset_class in {"commodity", "metal"}:
        return _last_bd(*_prev_month(year, month))   # approximation, Part 2 context only
    return None


def last_trading_day(root: str, year: int, month: int) -> pd.Timestamp | None:
    if root in US_TREASURIES:
        return pd.Timestamp(np.busday_offset(_last_bd(year, month).date(), -7, roll="backward"))
    if root in EUREX_BONDS:
        tenth = pd.Timestamp(year=year, month=month, day=10)
        return pd.Timestamp(np.busday_offset(tenth.date(), -2, roll="backward"))
    if root == "G":
        return pd.Timestamp(np.busday_offset(_last_bd(year, month).date(), -2, roll="backward"))
    return None


def roll_anchor_date(root: str, ticker: str, ref_date=None) -> tuple[pd.Timestamp | None, str]:
    """First notice day or last trading day, per the root's spec."""
    spec = SPECS.get(root)
    if spec is None:
        return None, "unsupported"
    _, year, month = parse_contract(ticker, ref_date=ref_date)
    fn = first_notice_day if spec.anchor == "fnd" else last_trading_day
    d = fn(root, year, month)
    return d, (spec.anchor if d is not None else "unsupported")
