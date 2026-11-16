# Task 3: Monthly Spending Prediction (`spending_30d`)

**Champion Method:** Leak-Free 5-Fold Bayes-Optimal Multi-Quantile Hurdle ($Q_{1 - 0.5/p}$)  
**Competition Metric:** Normalized Mean Absolute Error — $\text{NMAE} = \frac{1}{1 + \sum|y - \hat{y}| / \sum y}$

## Verified 5-Fold Nested Outer OOF Performance
| Model / Ablation Stage | Validation Protocol | OOF MAE (THB) | OOF NMAE |
| :--- | :--- | ---: | ---: |
| Fold-Training Median Baseline | 5-Fold Outer OOF | 10,495.48 | 0.500037 |
| Prior Frozen External Baseline | 5-Fold Nested OOF | 2,796.93 | 0.789609 |
| Prior Hybrid Hurdle + Meta-Stacker | 5-Fold Nested OOF | 2,760.47 | 0.791781 |
| Stage-2 Log-Linear Ridge Hurdle | 5-Fold Outer OOF | 2,842.19 | 0.786930 |
| LightGBM Quantile ($Q_{1-0.5/p}$) | 5-Fold Outer OOF | 2,716.87 | 0.794393 |
| XGBoost MultiQuantile ($Q_{1-0.5/p}$) | 5-Fold Outer OOF | 2,714.42 | 0.794541 |
| CatBoost MultiQuantile ($Q_{1-0.5/p}$) | 5-Fold Outer OOF | 2,702.42 | 0.795263 |
| **Promoted Champion (Nested BayesQ)** | **5-Fold Nested Outer OOF** | **2,702.65** | **0.795249** |

* **Per-Outer-Fold NMAE:** `[0.785353, 0.797148, 0.798009, 0.797133, 0.798254]` (`Mean +/- SD = 0.795179 +/- 0.005516`)
* **Paired Bootstrap 95% CI vs Log-Linear Ridge:** `+0.008319` (`[+0.007528, +0.009090]`)
* **Zero-Spender Alignment:** Train `48.12%` vs Test Predicted `49.57%` (`N = 25,889`)

## Key Architectural Innovations
1. **Deterministic Identity Cleaning (`99.0%` exact):** Reconstructs missing `historical_spending` via `total_transactions * avg_transaction_value` (or `avg_transaction_value` when `tx == 0`) before falling back to fold-local median `prev / hist` ratios.
2. **Fold-Local Missing Segment Recovery:** Trains an internal `LGBMClassifier` on known-segment training rows to supply soft probabilities `p_seg_0..3` for MCAR missing segments (`~92.2%` CV accuracy).
3. **Bayes-Optimal $L_1$ Quantile Evaluation:** Models $\varepsilon = \log(Y) - \hat{z}_{\text{Ridge}}$ across 9 conditional quantiles $[0.03, \dots, 0.51]$ and interpolates strictly at $q^*(p_i) = 1 - \frac{1}{2 p_i}$ with strict per-row monotonicity and internal 90/10 early-stopping splits.

## Reproducible Execution
Execute `task3_monthly_spending.ipynb` end-to-end; passing the promotion gate automatically writes verified outputs into `predictions/test_predictions.csv`, `predictions/oof_predictions.csv`, `submit/index.csv`, `assets/`, and `models/`.
