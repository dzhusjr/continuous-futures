"""Contract metadata and the roll thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

MONTH_CODES = dict(zip("FGHJKMNQUVXZ", range(1, 13)))

CALENDAR_OFFSET_BD = 3  # calendar rule rolls this many sessions before the anchor


class Spec(NamedTuple):
    root: str
    name: str
    asset_class: str
    exchange: str
    cycle: str
    anchor: str   # 'fnd' first notice day or 'ltd' last trading day
    depth: int


#   root   name                      class      exchange cycle      anchor depth
_TABLE = """
    TU   | 2y US Treasury Note     | bond      | CBOT   | quarterly | fnd | 2
    FV   | 5y US Treasury Note     | bond      | CBOT   | quarterly | fnd | 2
    TY   | 10y US Treasury Note    | bond      | CBOT   | quarterly | fnd | 2
    US   | US Treasury Bond        | bond      | CBOT   | quarterly | fnd | 2
    WN   | Ultra US Treasury Bond  | bond      | CBOT   | quarterly | fnd | 2
    RX   | Euro-Bund (10y)         | bond      | EUREX  | quarterly | ltd | 2
    OE   | Euro-Bobl (5y)          | bond      | EUREX  | quarterly | ltd | 2
    DU   | Euro-Schatz (2y)        | bond      | EUREX  | quarterly | ltd | 2
    OAT  | Euro-OAT (French 10y)   | bond      | EUREX  | quarterly | ltd | 2
    IK   | Euro-BTP (Italian 10y)  | bond      | EUREX  | quarterly | ltd | 2
    G    | Long Gilt               | bond      | ICE    | quarterly | fnd | 2
    SFR  | 3m SOFR                 | stir      | CME    | quarterly | ltd | 16
    ER   | 3m Euribor              | stir      | ICE    | quarterly | ltd | 16
    SFI  | 3m SONIA                | stir      | ICE    | quarterly | ltd | 16
    IR   | 90d Bank Bill           | stir      | ASX    | quarterly | ltd | 16
    CL   | WTI Crude               | commodity | NYMEX  | monthly   | fnd | 2
    CO   | Brent Crude             | commodity | ICE    | monthly   | ltd | 2
    QS   | Gasoil                  | commodity | ICE    | monthly   | ltd | 2
    XB   | RBOB Gasoline           | commodity | NYMEX  | monthly   | fnd | 2
    GC   | Gold                    | metal     | COMEX  | monthly   | fnd | 2
    SI   | Silver                  | metal     | COMEX  | monthly   | fnd | 2
    HG   | Copper                  | metal     | COMEX  | monthly   | fnd | 2
"""

SPECS: dict[str, Spec] = {}
for _line in _TABLE.strip().splitlines():
    _f = [c.strip() for c in _line.split("|")]
    SPECS[_f[0]] = Spec(_f[0], _f[1], _f[2], _f[3], _f[4], _f[5], int(_f[6]))

BOND_SPECS = {k: v for k, v in SPECS.items() if v.asset_class == "bond"}
US_TREASURIES = {"TU", "FV", "TY", "US", "WN"}
EUREX_BONDS = {"RX", "OE", "DU", "OAT", "IK"}


@dataclass(frozen=True)
class RollParams:
    """Thresholds on the front share s = OI_front / (OI_front + OI_next)."""
    start_threshold: float = 0.85
    roll_threshold: float = 0.50
    end_threshold: float = 0.15
    confirm_days: int = 2
    min_pair_oi: float = 1000.0


DEFAULT_ROLL_PARAMS = RollParams()
ACTIVITY_COLS = ["PX_LAST", "PX_VOLUME", "OPEN_INT"]
