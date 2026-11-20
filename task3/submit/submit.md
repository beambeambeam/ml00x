# Task 3 Current Submission (`submit/index.csv`)

- **Active Version:** **Run 22 / Run 18** (`Palapluem`, `ExactBayesL1V2` from `#57`)
- **Generating Notebook:** `task3/task3_monthly_spending.ipynb`
- **Required Artifacts:**
  - `task3/predictions/model-level/task3-exact-bayes-v2-bundle.npz`
  - `task3/data/train.csv`, `task3/data/test.csv`, `submission/sample_submission.csv`
- **Outputs & Metadata:**
  - `task3/predictions/oof_predictions.csv`, `task3/predictions/test_predictions.csv`
  - `task3/models/cv_metrics_summary.json`, `task3/models/artifact_sha256.json`, `task3/assets/ablation_benchmark.csv`
- **OOF Metrics:**
  - Pooled Nested 5-Fold Outer OOF NMAE = **`0.796377`** | MAE = **`2,683.96 THB`**
  - Per-fold mean ± SD = `0.796306 ± 0.005708` (`Δ vs Run 16 V3 0.795996 = +0.000381`)
- **Test Summary (`25,889` rows):** zeros `49.6%`, median `78.7571 THB`, max `300,000 THB`
- **`index.csv` SHA-256:** `3646bdd6dda1d6f7572c9e53addaed8e55f7aafcc2c1387aa9eb06c095c7647b`
