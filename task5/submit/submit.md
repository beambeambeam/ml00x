# Task 5 Current Submission (`submit/index.csv`)

- **Active Version:** **Run 22 / Run 18** (`Palapluem`, restored in PR `#70`)
- **Generating Notebook:** `task5/task5_account_security.ipynb` (self-contained)
- **Required Input:** `task5/data/test.csv` (`25,889` accounts × `123` columns)
- **Outputs:** `task5/predictions/test_predictions.csv`, `task5/assets/task5_anomaly_detection_diagnostics.png`
- **Detector Components & Unsupervised Checks:**
  - `1D Density-Gap CBLOF` (`1,808`) + `Soft-Max Robust Tail Score > 18.35` (`826`) + `Wide-Gap Box`
  - `Uniform-Box Latent Unstable-Behavior Recovery` (`3-Fam Platt`: `StrictUnflagged_AUC = 0.99030`, `F3-Bayes Thr = 0.10`, `+37` NaN accounts)
  - `Archetype 8 Joint 12D Hyper-Rectangle` (`erratic_combat_variance`: `+244` new accounts)
  - Dropped `session_duration_std__drift4` Gaussian noise artifact (`-51` FPs)
- **Test Distribution (`25,889` rows):** `0` (normal): `23,800` (`91.9%`), `1` (anomalous): `2,089` (`8.07%` / `8.1%`)
- **`index.csv` SHA-256:** `92253faba1f6102dd6871cd1c76f2684c322f478fdd2093b3f8d3c9031b9809e`
