# ADR 007 — Portfolio Correlation Check via LLM

**Status**: draft
**Date**: 2026-03-17
**Branch**: feat/llm-correlation-check

---

## Context

The bot was placing bets on highly correlated or mutually exclusive markets simultaneously.
Two concrete failure modes observed:

1. **Exhaustive-partition buckets** — e.g. "Elon Musk tweets 260-279" and "Elon Musk tweets
   280-299" in the same week. At most one can resolve YES. Betting YES on both is equivalent
   to buying partial coverage of a single event at inflated combined cost.

2. **Threshold ladders** — e.g. simultaneous NO bets on Crude Oil hitting $100, $105, $110,
   $130 by end of March. These are not independent: if oil hits $105 it necessarily hit $100,
   making the positions logically entangled and effectively over-leveraged on the same outcome.

The existing deduplication only covers:
- Same `market_id` (exact duplicate)
- Same `event_slug` (Polymarket's own event grouping)

This misses correlations across separately-grouped Polymarket events that share a common
underlying question.

Embedding-based similarity was considered but rejected: cosine distance catches lexical
overlap but cannot reason about logical implication (threshold ladders) or mutual exclusivity
(partition buckets). LLM reasoning is required.

---

## Change

### New struct: `CorrelationDecision` (`scanner/live.rs`)

```rust
pub struct CorrelationDecision {
    pub market_id: String,
    pub keep: bool,
    pub reason: String,
}
```

### New method: `LiveScanner::check_portfolio_correlation`

Single LLM call (temperature 0.0, deterministic) that receives:
- **Candidates**: list of `(id, question, side)` — all signals that passed every prior guard
- **Open bets**: list of `(question, side)` — current unresolved positions

Returns one `CorrelationDecision` per candidate. Defaults to `keep = true` on any parse
or network error (fail-open, never blocks bets silently).

The prompt instructs the model to REJECT a candidate only if it is:
- Logically correlated with an existing bet (e.g. threshold ladder)
- Mutually exclusive with an existing bet (e.g. partition bucket on the same period)
- Essentially the same event under a different title/framing

Independent candidates (different underlying questions) are always KEEP.

### Restructured bet loop: `cycles/bet_scan.rs`

Old flow (interleaved evaluate + place):
```
for signal in signals:
  for strat: evaluate → place_bet immediately
```

New flow (two phases):
```
Phase 1 — collect: for signal in signals, for strat: evaluate → collect accepted list
Phase 2 — filter:  one LLM call → CorrelationDecision per candidate
Phase 3 — place:   place survivors; log + notify TG for rejected ones
```

### Logging and notifications

- **Placed bet message**: appended with `✅ Correlation check: <reason>` from LLM.
- **Rejected by correlation**: `tracing::info!` + Telegram message listing the candidate
  question, side, and the LLM's rejection reason, plus the conflicting open bet it cited.

---

## Trade-offs

| | Pro | Con |
|--|-----|-----|
| Single batch call | One LLM call regardless of N candidates | Slightly more complex prompt/parse |
| Fail-open on error | Never silently blocks bets | A broken check passes through correlated bets |
| LLM vs embeddings | Understands implication/exclusivity | Slower, costs tokens |

---

## Conclusion

TBD after live observation. Success criterion: no more simultaneous bets on partition buckets
or threshold ladders for the same underlying outcome within a single scan cycle.
