# Continuous Futures & Roll Handling

Turning a series of individual, expiring futures contracts into a clean continuous
**price** (for returns) and a clean continuous **open interest** (for positioning) —
with a roll definition that is precise, testable, and portable across contracts.

Primary instrument: **TY** (10y US Treasury note). Second bond future, run with
nothing redesigned: **RX** (Euro-Bund). The definition is then swept across all
eleven bond roots in the file.

---

## Run it

```bash
pip install -r requirements.txt

# everything: tables and headline numbers (~15 s)
python scripts/run_analysis.py --data data/FUT_raw_data.csv

# the claims in the write-up, as assertions (113 tests, ~2 s)
python -m pytest tests -q
```

Outputs land in `outputs/`: `tables/*.csv` and `headline_numbers.json`.


## Deliverables

| File | What it is |
|---|---|
| `writeup/Writeup.pdf` | The two-page write-up (Part 3). |
| `TIME_AND_TOOLS.md` | Time spent and what AI tools were used for. |
| `scripts/run_analysis.py` | End-to-end run on the provided file. |

---

## The roll definition

Within one *cycle* — the span over which generic 1 maps to a single contract `C_k`
and generic 2 to `C_{k+1}` — let

```
s_t = OI_front(t) / ( OI_front(t) + OI_next(t) )
```

Three dates, each the first of **two consecutive sessions** meeting the condition:

| | condition |
|---|---|
| **roll start** | `s < 0.85` |
| **roll date** | `s < 0.50` ← the series changes contract at this settlement |
| **roll end** | `s < 0.15` |

A crossing only counts if `s` was above 0.85 earlier in the same cycle: a roll is a
*crossing*, not a *level*. Thresholds act on a bounded ratio, so the same numbers
run unchanged on a 3.9m-lot Treasury contract and a 150k-lot BTP contract.

Volume-crossover and calendar/first-notice rules are implemented too, and compared
rather than dismissed — the calendar rule agrees to within **one session on all
eleven bond roots**, which is the evidence that the OI rule finds the real event.

## The finding that drives everything else

The data provider's generic switches contract on the first session **after the
expiring contract's last trading day**. Liquidity leaves much earlier — at first
notice, where the contract has one.

| Group | Gap, OI roll → generic switch | Days generic 1 is the wrong contract |
|---|---|---|
| TU, FV (CBOT) | 26 sessions | 40–41% |
| TY, US, WN (CBOT) | 19 sessions | 30% |
| G Long Gilt (ICE) | 25 sessions | 38% |
| RX, OE, DU, OAT, IK (Eurex) | 4–5 sessions | 7% |

On TY the stub contract's open interest falls from 3.9m lots to 77 while generic 1
still points at it. For US Treasuries and Gilts, roughly **one observation in three**
of "the front generic" describes a contract that no longer holds the risk.

`roll.selector` fixes this: it tracks which generic actually holds the risk on each
day, so the continuous series rolls on the OI date and never reads the dead stub.

## Results

**Price** — each construction scored on the days *it* rolls, against a splice-return
benchmark that cannot be wrong (both legs of every return from the same contract):

| series | injected jump (TY) | RAS | p |
|---|---|---|---|
| provider generic 1, unadjusted | 44.7 bp RMS / 120 bp max | +0.92 | 0.0001 |
| difference (Panama) back-adjusted | 0.40 bp | −0.22 | 0.31 |
| ratio back-adjusted | 0.00 bp | −0.21 | 0.31 |

**Positioning** — the chosen series is whole-curve aggregate OI (`(OI1+OI2)/AGG` has
median exactly 1.00000 on every bond root, so the aggregate *is* the active pair):

| series | median \|Δlog\| at roll | calendar R² | corr. with chosen |
|---|---|---|---|
| generic-1 OI | 63% | 0.88 | **0.14** |
| curve aggregate OI | 2.2% | 0.13 | 1.00 |
| aggregate, deseasonalised | — | 0.0006 | 0.93 |

A conviction signal built on generic-1 open interest correlates **0.14** with the
correct one. It is not a noisy version of the right signal; it is mostly a
different, calendar-driven series.

## Layout

```
src/contfut/
  config.py       contract metadata and the roll thresholds
  data.py         loading, traded calendar, staleness filter  (all cleaning decisions documented here)
  calendars.py    ticker parsing, first notice / last trading day per exchange
  roll.py         the roll definition, three detectors, the selector
  continuous.py   splice / difference / ratio price; four OI constructions
  metrics.py      RAS + permutation test, injected jump, calendar R²
scripts/run_analysis.py
tests/            113 tests: invariants on real data, algebra on synthetic data, detector behaviour
```

## Data cleaning, in one place

1. **Forward-filled non-trading days are dropped.** A traded calendar is rebuilt per
   root: a date is a session iff *any* generic changed in (PX_LAST, VOLUME, OPEN_INT),
   with `NaN == NaN` counted as unchanged; weekends are dropped outright. TY: 2461
   calendar days → 1701 sessions, and the weekday exclusions are exactly the CME
   holiday list.
2. **Stale open interest and volume are dropped.** The provider carries a generic's OI
   forward when the underlying has not started reporting. Two patterns: at a generic
   switch the new generic 2 keeps the *previous* contract's OI (TY 2020-06-22:
   3,302,975 lots for seven sessions); and on Gilt/OAT/Schatz/Bobl generic-2 OI
   freezes for up to 47 sessions at a level comparable to generic 1 — far enough that
   the reported OI(2) *exceeds the whole curve aggregate*, which is how we know it is
   an artefact. A value is dropped when the ticker changed but the value did not, or
   when it is the 3rd member of a run of identical sessions. Front-month OI on
   TY/RX/US/OE never trips this test, so the filter costs nothing where the data is good.
3. **Missing OPEN_INT is kept as NaN, never filled.** It occurs only in rank ≥ 3 of a
   few roots in 2020–21 and in the back end of the STIR strips; there are zero NaNs in
   ranks 1–2 of any bond root.
4. **Prices are not cleaned.** CL settled at −37.63 on 2020-04-20; it is a real print,
   and it is the argument against ratio/log-based continuation in Part 2.
