# Task 5: Account Security & Bot Anomaly Detection

**Course:** CPE 342 Machine Learning (Semester 1/2026) — KMUTT  
**Instructor:** Dr. Boonyarit Changaival  
**Department:** Department of Computer Engineering, King Mongkut's University of Technology Thonburi  

---

## 1. Problem Formulation & Metric

- **Target Variable:** `is_anomaly` $\in \{0, 1\}$.
  - `0`: Normal Account Activity (~91.0%)
  - `1`: Account Security Anomaly / Bot Behavior (~9.0%)
- **Official Competition Metric:** Binary Classification $F_1$ Score / Balanced Accuracy:

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- **Dummy Sample Baseline (Predicting all 0):** $F_1 = 0.0000$
- **Our Ensembled Solution:** High-recall multi-detector security ensemble calibrated to test prior.

---

## 2. Methodology & Architecture

1. **Multi-Family Feature Extraction:**
   - Communication frequency, automated latency deviations, session entropy, device fingerprint variance, and transaction burst rates.

2. **Tri-Model Security Architecture:**
   - **Unsupervised Anomaly Detector:** Isolation Forest (`contamination=0.09`) measuring tree depth path length in high-dimensional feature space.
   - **Cost-Sensitive Supervised Classifier:** Balanced Random Forest with balanced subsampling.
   - **Gradient Boosted Detector:** XGBoost with `scale_pos_weight` tuned to the positive class imbalance.

3. **Empirical Prior Calibration:**
   - Composite anomaly score aggregated via weighted probability ranking.
   - Decision threshold locked at the 91.0th percentile ($0.7312$), yielding exactly 9.0% positive anomaly flags on the unlabelled test partition.

---

## 3. Test Prediction Distribution (25,889 records)

| Account Security Class | Sample Count | Percentage |
| :---: | :---: | :---: |
| `0` (Normal Account) | 23,559 | 91.0% |
| `1` (Security Anomaly) | 2,330 | 9.0% |
