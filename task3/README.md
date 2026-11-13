# Task 3: 30-Day Monthly Spending Prediction

**Team:** ML004  
**Course:** CPE 342 Machine Learning (Semester 1/2026) — KMUTT  
**Instructor:** Dr. Boonyarit Changaival  
**Department:** Department of Computer Engineering, King Mongkut's University of Technology Thonburi  

---

## 1. Problem Formulation & Metric

- **Target Variable:** `spending_30d` ($\ge 0.0$ THB).
- **Data Characteristics:** Zero-inflated continuous distribution (48.12% exact zero spending; observed target range $0$–$300{,}000$ THB).
- **Official Competition Metric:** Normalized Mean Absolute Error (NMAE)

$$
\text{NMAE} = \frac{1}{1 + \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}}
$$

- Baseline (Predicting Global Median): $\text{NMAE} = 0.5000$
- Challenge Target: $\text{NMAE} \ge 0.7500$
- **Previous Frozen Benchmark (Nested 5-Fold OOF):** $\text{NMAE} = 0.7896088679$ (MAE: $2{,}796.9345$ THB; $N=104{,}000$).
- **Promotion status (2026-11-10): Not promoted.** Champion reference: $0.7917806249$; strict gate: $>0.791781$ plus a positive same-row 95% paired-bootstrap CI. The available scratch bundle recomputes to NMAE $0.7917661927$ (MAE $2{,}760.7124$ THB); its evaluation is not valid nested CV, the champion OOF vector is absent, and its test array differs from current CSV artifacts.

---

## 2. Methodology & Key Breakthroughs

Standard single-stage regression models suffer from gradient conflict between the point mass at zero ($48.12\%$) and the heavy-tailed whale distribution. Conversely, pure log-multiplier models suffer from exponential error amplification on high-value whales. The candidate architecture under evaluation combines a **3-Family Scale-Relative Log-Multiplier Two-Stage Hurdle** with **Direct Raw-Scale L1 Regressors** and a **Level-2 `LGBM(MAE)` Meta-Stacker**:

### 2.1 High-Fidelity Spending Reconstruction (Fold-Local Empirical Proxy)
Empirical analysis of 86,961 non-null training records reveals that:

$$
\text{historical\_spending} \approx \text{total\_transactions} \times \text{avg\_transaction\_value}
$$

holds within $0.01$ THB for **99.00%** of records ($86{,}093 / 86{,}961$). Exactly 868 records ($1.00\%$) deviate by $> 1.00$ THB (median discrepancy $857.31$ THB). Within each validation fold, the complementary ratio `prev_to_hist` and feature medians are estimated strictly on that fold's training partition to prevent cross-fold leakage.

### 2.2 Hybrid Two-Stage Hurdle + Direct Raw-Scale L1 Stacking
1. **Stage 1 Spender Gatekeeper ($P(\text{spending}_{30\text{d}} > 0)$):** Tri-model probability ensemble of `LGBMClassifier`, `CatBoostClassifier`, and `XGBClassifier`, calibrated via fold-local `IsotonicRegression`.
2. **Stage 2 Scale-Relative Log-Multiplier Regressors:** Trained on positive spenders using target $z_i = \ln(1 + y_i / \max(\text{prev\_clean}_i, 100))$ across `LGBMRegressor(regression_l1)`, `CatBoostRegressor(MAE)`, and `XGBRegressor(reg:absoluteerror)`.
3. **Direct Raw-Scale L1 Regressors:** Trained on the full cohort directly in raw THB scale (`LGBMRegressor(mae)`, `CatBoostRegressor(MAE)`, `XGBRegressor(reg:absoluteerror)`) so leaf medians anchor extreme whale predictions without exponential amplification.
4. **Level-2 `LGBM(MAE)` Meta-Stacker & Nested Hurdle Gate:** Non-linearly combines Hurdle and Raw-Scale L1 predictions with fold-local probability gating $\mathbb{I}[P_{\text{cal}, i} \ge \theta^*]$.

---

## 3. Previously Reported Candidate Scores (Not Promotion-Verified)

| Architecture | Reported MAE (THB) | Reported OOF NMAE | Audit status |
| :--- | :---: | :---: | :--- |
| 1. Previous Frozen Two-Stage Hurdle Baseline | 2,796.9345 | 0.7896088679 | Superseded baseline |
| 2. Direct Raw-Scale L1 Ensemble (LGBM + CB + XGB) | 2,851.7079 | 0.7863688920 | Component |
| 3. Level-2 LGBM (MAE) Meta-Stacker | 2,788.4938 | 0.7901105291 | Exploratory; not nested |
| 4. 3-Family Two-Stage Log-Multiplier Hurdle | 2,769.6809 | 0.7912309444 | Exploratory; not nested |
| 5. Hybrid Hurdle + Raw-Scale L1 Ensemble | 2,762.9464 | 0.7916327989 | Exploratory; not nested |
| **6. Gated Super-Hybrid + Level-2 Meta-Stacker (reported candidate)** | **2,760.4707** | **0.7917806249** | **Not promoted; protocol unverified** |

---

## 4. Execution & Artifacts

Current predictions remain uncommitted candidate artifacts. Do not treat them as promoted until a true nested run, same-row champion comparison, and positive 95% paired CI are available.

- **Jupyter Notebook:** `task3_monthly_spending.ipynb` (22 cells: 12 Markdown, 10 Code; AST verified).
- **Test Set Predictions:** `predictions/test_predictions.csv` (25,889 rows aligned with `test.csv`).
- **Official Submission:** `submit/index.csv` (25,889 rows: `ANS00001`–`ANS25889`, $48.63\%$ zeros, median $87.00$ THB, max $344{,}055.47$ THB).
- **Report Assets:** 300-DPI diagnostic figures and `ablation_benchmark.csv` in `assets/`.
