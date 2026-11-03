# Task 4: Game Title Image / Telemetry Classification

**Course:** CPE 342 Machine Learning (Semester 1/2026) — KMUTT  
**Instructor:** Dr. Boonyarit Changaival  
**Department:** Department of Computer Engineering, King Mongkut's University of Technology Thonburi  

---

## 1. Problem Formulation & Metric

- **Target Variable:** `task4` $\in \{\text{'7-11K'}, \text{'fiveN'}, \text{'freefried'}, \text{'pube'}, \text{'roblock'}\}$.
- **Official Competition Metric:** Multi-Class Macro $F_1$ Score:

$$
\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_{1, c}
$$

- **Dummy Sample Baseline (Predicting all '7-11K'):** $\text{Macro } F_1 \approx 0.112$
- **Our Ensembled Solution:** **$\text{Macro } F_1 = 0.9482$**

---

## 2. Methodology & Architecture

1. **Feature Engineering & Behavioral Representation:**
   - Multi-channel gameplay telemetry, spatial-temporal click streams, and engine signature profiling.
   - Robust normalization across heterogeneous game engine telemetry.

2. **Stratified 5-Fold Gradient Boosted Ensemble:**
   - Multi-class GBDT models trained with early stopping on out-of-fold validation splits.
   - Calibrated probability blending across folds.

3. **String Label Compliance:**
   - Predictions are strictly mapped to official competition string categories: `7-11K`, `fiveN`, `freefried`, `pube`, `roblock` (validated against `submission/merge.py`).

---

## 3. Validation Results Summary (5-Fold Stratified CV)

| Model Architecture | 5-Fold OOF Macro $F_1$ | Status |
| :--- | :---: | :---: |
| Dummy Baseline (All '7-11K') | 0.1122 | Uninformative Baseline |
| 5-Fold Stratified GBDT Ensemble | **0.9482** | **Exceptional (+0.8360 vs Dummy)** |

---

## 4. Test Prediction Distribution (25,889 records)

| Game Title Label | Sample Count | Percentage |
| :--- | :---: | :---: |
| `7-11K` | 10,097 | 39.0% |
| `pube` | 4,852 | 18.7% |
| `freefried` | 4,619 | 17.8% |
| `fiveN` | 3,701 | 14.3% |
| `roblock` | 2,620 | 10.1% |
