# Time spent and use of AI tools

**Total: roughly 5 hours.**

| Block | Time | What it covered |
|---|---|---|
| Data | ~1h | Reading the file, reconstructing the traded calendar, finding and characterising the two forward-fill / staleness traps. This took the longest and was the least expected. |
| Roll definition | ~1.5h | Choosing open interest over volume and calendar, the crossing-not-level refinement, and the comparison against `FUT_CUR_GEN_TICKER` (Part 1.2). |
| Continuous price and OI | ~1h | The three price constructions and the algebra that makes the invariants exact; the four OI candidates and the aggregate-equals-pair check. |
| Metric and cross-root sweep | ~45m | RAS, the permutation test, calendar R², and running all 11 bond roots unchanged (Part 1.6). |
| Part 2 and write-up | ~45m | STIR strip evidence, the crude contrast, and the two-page write-up. |

## AI tools

AI (Claude) was used for:

- **Boilerplate**: pandas plumbing, the HTML/CSS of the write-up, and the mechanical parts of the test suite.
- **Lookups**: exchange contract specifications — CBOT Treasury first notice day and last trading day, Eurex delivery day conventions, ICE Long Gilt notice rules. **Every one of these was verified against the data** before being used: the implied last trading days reproduce the observed `FUT_CUR_GEN_TICKER` switch dates to the session on TY, RX and G, which is why the calendar comparator in `calendars.py` can be trusted at all.

AI was **not** used for the parts that the task says should be my own thinking:

- the roll definition and the choice of open interest over volume and calendar rules;
- the data-cleaning decisions (which days are sessions, which values are stale, what to do about missing OI and the negative crude print);
- the design of the robustness metric (RAS, the own-event scoring rule, calendar R²) and the decision to benchmark everything against a splice-return series;
- the Part 2 argument about STIRs and the conclusions in the write-up.
