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

AI (Claude) was used for **boilerplate code, documentation and tests**.

It was **not** used for the analysis itself: the roll definition, the data-cleaning
decisions, the design of the robustness metric, and the Part 2 reasoning and
conclusions are my own.
