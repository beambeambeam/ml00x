# Task audit and selected improvement

Audit date: 2026-11-21. Baseline: `main` at `8e9eede57aa5d4b588cc5f5861c1e88807db668e`.

Only Task 2 changes. Scores below come from repository records; metrics across tasks are not comparable.

| Task | Current state | Improvement opportunity and selection |
| --- | --- | --- |
| 1 | Experiment 005 account-cohort and skill classifier. Recorded OOF F2 0.840965. | Fresh improvement already merged. Further gains require new model validation; cohort ranges and blend weight used exploratory labeled-data analysis. |
| 2 | Run 22 probability models. Recorded affine-rule OOF macro F1 0.759316, but executable export uses a different rule. | Selected: the affine export is Markdown, while the executable export drops beta. Restore the evaluated rule and test the actual export. |
| 3 | ExactBayesL1V2. Recorded OOF NMAE 0.796377. | Dependencies are undeclared and the active notebook replays a saved bundle. Reproducible base-model training needs work; claimed near-ceiling performance leaves less clear scoring headroom. |
| 4 | Run 22 restored after Run 23 regressed on Kaggle. Recorded OOF macro F1 0.991174. | Further model training needs GPU compute. The recent OOF improvement failed to transfer to Kaggle, so validation and distribution shift need attention. |
| 5 | Archetype rules plus recovery for missing behavior stability. | No true anomaly labels. Masked-feature checks are available, but the reported expected F3 is not a measured competition score. New flags need stronger evidence. |

Sources: `task{N}/submit/submit.md`, root `submit.md`, active task notebooks and scripts,
`task3/pyproject.toml`, and `task2/notebook/tried/README.md`.

## Verified Task 2 change

The old multiplier-only rule reproduces all 25,889 baseline submission labels.
The corrected rule is `argmax(alpha * probability - beta)` in both validation and export.
No model, feature, or split changes. Test probabilities do not enter parameter fitting.

| Measure | Old executable export | Corrected affine export |
| --- | ---: | ---: |
| Pooled OOF macro F1 | 0.753157591 | 0.759315844 |
| Pooled OOF weighted F1 | 0.763170897 | 0.771084821 |
| Test labels changed | 0 | 1,622 of 25,889 |

All five postprocessing folds improve. A 1,000-sample paired row bootstrap gives
a 95% macro-F1 gain interval of `[+0.004653, +0.007692]`.

This reuses frozen OOF probabilities. It is not a fully nested experiment:
base models behind other OOF folds can have seen the held-out postprocessing
fold's labels. The bootstrap does not measure model-training or selection
uncertainty. The NPZ has no row IDs, so original row ordering is assumed;
the replay records input hashes and checks keyed-model label alignment.
The corrected file has no Kaggle score yet.

## Reproduce

From `task2/`:

```bash
uv run python run_bayes_export.py
uv run python -m pytest tests/test_bayes_decision.py
```

Add `--promote` to the replay command to write `submit/index.csv` after the
five-fold and bootstrap gate passes. The model JSON stores stack weights,
alpha, beta, per-fold parameters, metrics, and input hashes. The notebook
uses the same decision function and has one executable export.

The complete notebook was executed in an isolated output directory. Its
submission matches the replay file byte for byte; see `notebook-verification.json`.
