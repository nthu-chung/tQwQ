# tQwQ — a Taiwan-equity backtest engine that tries to stop you lying to yourself

Most of the effort in this repository is not spent finding signals. It is spent
building gates that **refuse to produce a number** when the number would be
wrong. That is the whole thesis: on two years of daily data, a backtest is far
more likely to be broken than to be profitable, and a broken backtest looks
exactly like a good one.

This repo ships the machinery. It deliberately ships **no working strategy** ---
see [What is not here](#what-is-not-here).

## Why a whole repo for this

Every gate below exists because something silently produced a plausible, wrong
result. Not hypothetically --- each one has a specific incident behind it.

| Gate | What it caught |
|---|---|
| Price integrity is fail-closed | An unadjusted corporate action was being read as a real ~-73.6% loss, which then triggered a hard stop and changed which exit rule looked "best" |
| Return convention must match | Stock series were dividend-adjusted while the benchmark was a price index --- a systematic fake excess of ~2.86pp/year, positive in every one of 12 years |
| Listed-common-stock whitelist | Emerging-board stocks (no ±10% daily limit) leaked into the pool; their share of |daily move| > 10.5% was ~100x that of the main board, and momentum factors hunt exactly those |
| Dense panel for factors | `rolling(20)` on a sparse panel means "20 **rows**", which for an intermittent universe member spans 60+ calendar days |
| Ranking population is bound | Cross-sectional ranks were computed over a population that included stocks not buyable that day |
| Evaluation window upper bound | An in-sample equity curve overflowed the split point and scored part of the out-of-sample segment |
| Phase sweep, not one path | Changing only the rebalance start day moved Sharpe across the whole plausible range; the phase spread was about the size of the signal effect itself |

None of these raise a warning. Warnings get scrolled past. They raise.

## The two ideas that shape everything else

**1. The engine enforces what the market enforces. Nothing else.**

T+1 fills, ±10% limits snapped to legal tick, disposition-period entry bans,
1,000-share lots, a real cash ledger. Those are facts you cannot opt out of.
"Should the 20-day average be above the 60-day" is an *opinion*, and opinions
belong to a strategy where they are declared, parameterised, and hashed. An
earlier version had a moving-average gate hardcoded in the shared signal
builder, so every hypothesis silently inherited a rule nobody had declared ---
and it made whole classes of hypothesis (anything that buys weakness)
structurally untestable.

**2. Identity is two-layered, and the holdout is spent exactly once.**

```
strategy_rule_hash      which rules are we testing
evaluation_run_hash     how did we evaluate them
```

If a global config flag can change what you buy, then two runs with the same
`strategy_rule_hash` can trade different stocks --- and nothing errors. So the
out-of-sample segment is not merely "the later data": it is **segment-bounded at
the data layer** (the loader will not hand it to you), released once, through an
append-only hash-chained ledger, with a registry for windows you must declare
already-dirty. Changing a parameter gives you a new hash; it does not give you
back data you have already seen.

## Point-in-time, in two layers

```
month M candidate pool   built ONLY from the complete month M-1
                         (using M's own turnover to pick M's pool is
                          look-ahead wearing a calendar as a disguise)
        ↓
daily membership         top-N by trailing 20-day average turnover,
                         recomputed every signal day inside that month's pool
        ↓
dense panel              non-member rows kept, so ts_ operators see a
                         continuous per-stock series
        ↓
selection                membership applied here, not earlier
```

The split between those last two steps is the part people get wrong, including
us: filter first and every rolling window quietly changes meaning.

## Taiwan-specific rules that a generic backtester will get wrong

- **Tick size is price-dependent** and orders must snap to a legal price.
- **±10% limits**, with newly listed stocks exempt for their first days.
- **Disposition periods** (`處置`): matched every ~2 minutes with full
  pre-collection of cash/securities. Liquidity is materially different, so entry
  is banned. Exchange-level data quality differs --- one venue publishes actual
  disposition records, the other has to be derived from attention lists, and the
  code says which is which rather than pretending they are the same.
- **Lots are 1,000 shares.** At retail capital this often means the top-ranked
  name is not buyable as a whole lot, which is a portfolio-construction fact,
  not a rounding detail.
- **Adjusted vs as-traded prices are different questions.** Backtests need
  adjusted; a human reading a candidate list needs the number their broker
  shows. Both are carried, explicitly labelled.

## What is not here

**No working strategy, and that is deliberate.**

`strategies/` contains exactly one hypothesis: `h3_short_reversal` --- "buy what
fell the most". It is registered as a **control arm**, not a candidate. Its
documented purpose is to fail: if a deliberately weak reversal rule starts
producing an attractive Sharpe, the thing to distrust is the data or the
pipeline, not the market.

That makes it the most useful strategy to publish. It exercises the entire path
--- `score()` → validator → position policy → event engine → artifacts → audit
--- while being impossible to mistake for research.

The repo also carries **no performance figures for any strategy**. Numbers that
do appear (a passive benchmark's Sharpe, the price-index vs total-return-index
gap) are market facts or descriptions of bugs, kept because they are what make
the gates legible.

## Quickstart

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Offline: no token needed, all HTTP mocked
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# Secret / data-artifact / required-docs gate
PYTHONPATH=. .venv/bin/python preflight.py

# End-to-end on synthetic data
PYTHONPATH=. .venv/bin/python -m research.golden_path \
  --strategy h3_short_reversal --fixture synthetic \
  --capital research --output-dir /tmp/runs
```

Real market data needs `FINMIND_TOKEN` in the environment (never in the repo ---
`preflight.py` enforces that). Conventions: always `.venv/bin/python`, tests are
`unittest`, not pytest.

## Reading order

| Document | What it answers |
|---|---|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Module responsibilities and the invariants each one owns |
| [RESEARCH_OPERATING_PROTOCOL.md](./RESEARCH_OPERATING_PROTOCOL.md) | How a claim is allowed to be made |
| [DATA_SOURCES.md](./DATA_SOURCES.md) | What free Taiwan data actually gives you, and where it lies |
| [TAIWAN_MARKET_RULES.md](./TAIWAN_MARKET_RULES.md) | Limits, ticks, lots, disposition |
| [STRATEGY_REGISTRY.md](./STRATEGY_REGISTRY.md) | What was tried, what failed, and the platform-level lessons |
| [AGENTS.md](./AGENTS.md) | Working rules, including the seven traps that produce fake results |

## Disclaimer

Research and educational use. Nothing here is investment advice, and nothing
here has been shown to have an edge. See [DISCLAIMER.md](./DISCLAIMER.md).
