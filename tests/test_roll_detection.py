"""Behavioural tests for the roll detector itself."""
import numpy as np
import pandas as pd

from contfut.config import RollParams
from contfut.roll import first_sustained


def s(vals, start="2024-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_single_day_glitch_does_not_trigger():
    x = s([.99, .98, .10, .97, .96, .40, .30, .20])
    # the lone .10 must be ignored; the roll is the .40/.30 run
    assert first_sustained(x, 0.5, confirm=2) == x.index[5]


def test_confirm_one_would_have_been_fooled():
    x = s([.99, .98, .10, .97, .96, .40, .30])
    assert first_sustained(x, 0.5, confirm=1) == x.index[2]


def test_threshold_never_crossed_returns_none():
    assert first_sustained(s([.99, .98, .97]), 0.5, confirm=2) is None


def test_nans_are_skipped_not_treated_as_crossings():
    x = s([.99, np.nan, np.nan, .95, .40, .30])
    assert first_sustained(x, 0.5, confirm=2) == x.index[4]


def test_tail_shorter_than_confirm_window_still_counts():
    x = s([.99, .98, .40])
    assert first_sustained(x, 0.5, confirm=2) == x.index[2]


def test_params_are_ordered():
    p = RollParams()
    assert p.start_threshold > p.roll_threshold > p.end_threshold
