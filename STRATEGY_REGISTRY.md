# Strategy registry

A research state ledger. Not investment advice, and not a performance
leaderboard.

## Scope of this document

**Published:** which directions were tried, which were falsified, and *by what
mechanism* they failed. Plus the platform-level traps that produced fake results.
Those are the useful part --- a failure record stops the next person redoing the
work.

**Not published:** performance figures for any strategy, and factor definitions
that are still under evaluation. Those belong to the operator; this repo
publishes the **engine**.

So: **do not infer from this document which directions work.** Absence is not
evidence. A direction not listed here may have been tried and kept private, or
never tried at all.

## Status vocabulary

| Status | Meaning |
|---|---|
| `control arm` | Deliberately expected to lose. Exercises the pipeline; not a candidate. |
| `active hypothesis` | Rules and causal timing defined, worth testing on qualified data. Not proven alpha. |
| `blocked` | A researchable hypothesis, but a data or implementation integrity gap makes performance unusable. |
| `rejected` | Failed or redundant in relative comparison on available data. Not pursued further unless the data or definition materially changes. |
| `withdrawn` | Removed from this repository. Any holdout it consumed stays declared. |

---

## Registered

| ID | Direction | Status | Why it is here |
|---|---|---|---|
| `h3_short_reversal` | Buy the largest N-day decliners | `control arm` | **Expected to fail.** Existing evidence on this market and window is that weakness does not mean-revert on a horizon a swing strategy can hold. So this is the pipeline's canary: if a deliberately weak reversal rule produces an attractive Sharpe, distrust the data or the pipeline before believing the market changed. Single-factor, needs only `close`, ~40 lines --- which is what makes it a good end-to-end vehicle. |

That is the whole list, deliberately. See "Scope" above.

---

## Platform-level gates, and the incident behind each

These are the transferable part. Each one was written *after* something produced
a plausible wrong answer, and each one raises rather than warns --- warnings get
scrolled past.

### P0 — Unadjusted prices manufacture trades

A corporate-action gap in an unadjusted series was read as a real ~-73.6%
single-day loss. That tripped a hard stop, which changed which exit rule looked
best in a comparison. The bug did not just add noise; it changed the conclusion.

Implementation: unadjusted prices raise. A self-built adjustment is scanned for
residual breaks and blocked if any remain. **The break scan is a diagnostic, not
a clearance** --- a 3--5% dividend gap sits inside the ±10% band and is
structurally invisible to it, so "0 hits" does not mean "clean".

### P1 — Only listed common stock may enter the pool

A universe helper accepted a `market_type` argument and never used it, so its
real rule was "4 digits, not starting with 00". Emerging-board, DR, and
innovation-board issues all match that shape.

Why this is Sharpe-inflating rather than merely untidy: **emerging-board stocks
have no ±10% daily limit.** Their share of days with |return| > 10.5% was roughly
100x the main board's. Momentum factors hunt precisely those names, so the bias
has a direction. Liquidity screens do not save you --- the largest such name sat
inside the top-300 by turnover.

### P2 — Benchmark and stock series must share a dividend convention

Stock series were dividend-adjusted (the adjustment is equivalent to reinvesting
at the ex-date) while the benchmark was a **price** index. Measured over a
two-year window: about **2.86pp/year** of fake excess, and the sign was the same
in every one of 12 back-years. That magnitude lands squarely in
"looks-like-a-small-alpha" territory, which is why a warning was useless.

Implementation: convention is derived, recorded for both series, and a mismatch
raises.

### P3 — Survivorship is partly, not fully, solved

The monthly candidate pool is point-in-time and includes names that later
delisted. But **price coverage for delisted names is incomplete**, so the
overall `survivorship_free` flag stays `False`. Fixing a gate is not the same as
proving a strategy.

### P4 — Rebalance timing luck is not a rounding error

Take one signal set and change only which weekday the weekly rebalance starts.
Sharpe moved across essentially the whole plausible range, and the spread across
equivalent phases was about the size of the signal effect itself.

Therefore: **reporting a single phase is choosing a path.** Every formal result
runs all equivalent phases and reports median and worst --- never the best.
There is exactly one sweep implementation and an AST guard forbids a second.

### P5 — High IC does not mean a good backtest

Information coefficient measures rank correlation across the whole
cross-section. A strategy only buys the tail. We measured a case where the
highest-IC single factor lost to a lower-IC one in backtest. **Factor scans
decide what to investigate, never what to use.**

### P6 — The benchmark is a passive portfolio, not zero

Equal-weight buy-and-hold over the same daily-eligible universe already produces
a respectable Sharpe on this window. Beating zero is not alpha; beating the
basket you could have held instead is. Early reports claimed "beat the market"
partly from P2's convention gap and partly from comparing against nothing.

Also: the benchmark's *population* is part of the claim. Computing it over the
dense panel (which is mostly non-members by design) means benchmarking against a
basket you could not have held --- correcting that flipped the sign of an excess
return.

---

## Architectural decision: the trend gate is a strategy's opinion

`trend_ok = (MA20 > MA60) ∧ (MA60 slope > 0) ∧ (close > MA60)` began life as one
legacy screener's rule. During a layering refactor it moved into the shared
signal builder, and from there every hypothesis inherited it unconditionally.

Three separate problems, in increasing order of seriousness:

1. **Nobody declared it.** It appeared in no parameter table.
2. **It made identical rule hashes trade differently.** The switch was global
   config, so it entered evaluation-run identity rather than strategy-rule
   identity --- the exact thing the two-layer scheme exists to prevent.
3. **It made a whole class of hypothesis untestable.** Anything that buys
   weakness is, by construction, below its moving averages. A "deep drawdown"
   hypothesis had only ~11.8% of its intended sample survive the gate --- about
   one name per day, not enough to fill a slot. The hypothesis was not refuted by
   evidence; it was rewritten by infrastructure.

Resolution: it is now a per-strategy parameter that enters the rules hash, with
the default preserving prior behaviour (flipping a default would silently
redefine every existing strategy --- that should be a conclusion from re-testing,
not a side effect of a refactor). Declaring the gate without the column present
is now fail-closed instead of silently skipped.

The engine holds no view on trend at all; `tests/test_engine_has_no_trend_opinion.py`
uses an AST scan to keep it that way.

---

## Discipline that applies to every result here

1. **Absolute performance needs qualified prices.** Under unadjusted prices,
   returns, exit-rule comparisons, and IS/OS choices are all unverified.
2. **Compare against a passive benchmark, not zero** (P6).
3. **Run every equivalent phase; report median and worst** (P4).
4. **Factor scans choose what to investigate, not what to use** (P5).
5. **The out-of-sample segment is spent once.** Parameters may only be chosen on
   in-sample. Release goes through the single-holdout protocol and lands in an
   append-only ledger.
6. **Moving a strategy elsewhere does not un-spend its holdout.** There is one
   ledger; otherwise changing repository would launder "we already looked".

## Related

- [AGENTS.md](./AGENTS.md) — working rules ｜ [ARCHITECTURE.md](./ARCHITECTURE.md) — module map
- [DATA_SOURCES.md](./DATA_SOURCES.md) — what the free data actually gives you
- [RESEARCH_OPERATING_PROTOCOL.md](./RESEARCH_OPERATING_PROTOCOL.md) — how a claim may be made
- [TAIWAN_MARKET_RULES.md](./TAIWAN_MARKET_RULES.md) — limits, ticks, lots, disposition
