# Architecture Decision Records — Model

This directory tracks decisions and changes made to the prediction model,
including the motivation, what changed, and measured before/after results.

## Format

Each ADR is a numbered markdown file: `NNN-short-title.md`

Sections:
- **Status**: draft | active | superseded
- **Context**: why this decision was needed
- **Change**: what was modified
- **Metrics Before / After**: CV and live results
- **Conclusion**: what we learned

## Index

| # | Title | Status | Date |
|---|-------|--------|------|
| [001](001-baseline.md) | Baseline model audit | active | 2026-03-15 |
| [002](002-fix-training-inconsistencies.md) | Fix training/live inconsistencies | complete | 2026-03-15 |
| [003](003-fresh-data-fetch.md) | Fresh data fetch with deployment-window filter | complete | 2026-03-15 |
| [004](004-online-learning.md) | Online learning: feature store + warm-start retraining | draft | 2026-03-15 |
| [005](005-feature-improvements.md) | Temporal features, target encoding, SHAP analysis | active | 2026-03-15 |
