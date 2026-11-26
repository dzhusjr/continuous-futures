import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contfut.continuous import continuous_oi, continuous_price
from contfut.data import build_panel, load_raw
from contfut.roll import detect_rolls, roll_flags, selector

DATA = ROOT / "data" / "FUT_raw_data.csv"


@pytest.fixture(scope="session")
def raw():
    if not DATA.exists():
        pytest.skip(f"data file not found at {DATA}")
    return load_raw(str(DATA))


@pytest.fixture(scope="session", params=["TY", "RX", "G", "US", "OAT"])
def built(raw, request):
    root = request.param
    panel = build_panel(raw, root)
    rolls = detect_rolls(panel)
    sel = selector(panel, rolls)
    flags = roll_flags(panel, rolls)
    cp, gaps = continuous_price(panel, rolls, sel)
    oi = continuous_oi(panel, sel)
    return dict(root=root, panel=panel, rolls=rolls, sel=sel, flags=flags,
                cp=cp, gaps=gaps, oi=oi)
