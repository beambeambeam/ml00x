# Task 2: Player Segmentation Multi-Class Classification

**Course:** CPE 342 Machine Learning (Semester 1/2026) — KMUTT  
**Instructor:** Dr. Boonyarit Changaival  
**Department:** Department of Computer Engineering, King Mongkut's University of Technology Thonburi  

---

## 1. Problem Formulation & Metric

- **Target Variable:** `segment` $\in \{0, 1, 2, 3\}$.
  - `0`: Casual Players (~40.6%)
  - `1`: Core Players (~26.4%)
  - `2`: Hardcore Players (~19.5%)
  - `3`: Whale / VIP Players (~13.5%)
- **Official Competition Metric:** Multi-Class Macro $F_1$ Score:

$$
\text{Macro } F_1 = \frac{1}{C} \sum_{c=0}^{C-1} F_{1, c}
$$

- **Dummy Sample Baseline (Predicting all 0):** $\text{Macro } F_1 \approx 0.144$
- **Our Ensembled Solution:** **$\text{Macro } F_1 = 0.7478$**

---

## 2. Methodology & Architecture

1. **Feature Engineering Pipeline:**
   - Session duration, in-game transaction volume, engagement cadence, progression ratios, and recency-frequency telemetry.
   - Robust scaling and imputation for zero-variance features.

2. **Multi-Model 5-Fold Stratified Ensemble:**
   - **LightGBM:** `objective='multiclass'`, `num_class=4`, learning rate $0.03$, max depth $7$.
   - **XGBoost:** `objective='multi:softprob'`, learning rate $0.03$, colsample $0.8$.
   - **Blend:** Probability weighted combination ($0.55 \times \text{LGBM} + 0.45 \times \text{XGBoost}$).

3. **Coordinate-Ascent Class Multiplier Optimization:**
   - To maximize Macro $F_1$ across imbalanced segments, we optimize class decision weights $\mathbf{w}^* = [0.81, 1.03, 1.00, 0.98]$ on out-of-fold probability vectors before final `argmax`.

---

## 3. Validation Results Summary (5-Fold Stratified CV)

| Model Architecture | 5-Fold OOF Macro $F_1$ | Status |
| :--- | :---: | :---: |
| Dummy Baseline (All 0) | 0.1444 | Uninformative Baseline |
| Standalone XGBoost | 0.7405 | High Performance |
| Standalone LightGBM | 0.7454 | High Performance |
| Blend + Class Multiplier Optimization | **0.7478** | **Best Performance (+0.6034 vs Dummy)** |

---

## 4. Test Prediction Distribution (25,889 records)

| Segment | Sample Count | Percentage |
| :---: | :---: | :---: |
| `0` | 10,516 | 40.6% |
| `1` | 6,822 | 26.4% |
| `2` | 5,052 | 19.5% |
| `3` | 3,499 | 13.5% |
