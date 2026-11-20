# Task 3 submission (`submit/index.csv`)

> Agents keep this file. Rules: `AGENTS.md` → Submission history. Newest entry first. Never edit or delete old changelog entries.

## Current

- **sha12:** `3646bdd6dda1` (since 2026-11-19, `5acf8b8`, #57)
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

## Changelog

### 2026-11-19 – `3646bdd6dda1` – #57

- Commit: `5acf8b8` by Wisit Suwannao
- Prev: `3cd41be6c7f7` | Δrows vs prev: 13,063 (50.5%)
- Why: feat: Run 18 100% Pure-Train DGP Elevation across Tasks 1-5 (#57)
- Merged: `3f236f5e932e` | Kaggle public: 0.85908

### 2026-11-16 – `3cd41be6c7f7` – #50

- Commit: `c4b21da` by Wisit Suwannao
- Prev: `b96e2df2c4b4` | Δrows vs prev: 13,076 (50.5%)
- Why: feat(run16): upgrade Task 3 V3+MPS & Task 5 4-Fam Raw-NaN + Run 15 T1/T2/T4 (#50)
- Merged: `d57456bd72f3` | Kaggle public: 0.83969

### 2026-11-15 – `b96e2df2c4b4` – #46

- Commit: `c7dc931` by Wisit Suwannao
- Prev: `192b8145f303` | Δrows vs prev: 13,070 (50.5%)
- Why: feat(run14): upgrade Tasks 2 & 3 zero-leakage nested ensembles + Task 4 4xViT+SWA stack (#46)
- Merged: `1362e3d02daa` | Kaggle public: 0.83904

### 2026-11-14 – `192b8145f303` – #44

- Commit: `997bcd2` by Wisit Suwannao
- Prev: `48584320cb18` | Δrows vs prev: 13,064 (50.5%)
- Why: feat(run12): upgrade all 5 tasks + Task 3 dual-grid Bayes quantile hurdle (#44)
- Merged: `1727f0c63c80` | Kaggle public: 0.81773

### 2026-11-13 – `48584320cb18` – #40

- Commit: `9bc7502` by Wisit Suwannao
- Prev: `6192ed08e604` | Δrows vs prev: 25,889 (100.0%)
- Why: feat: Run 11 SOTA leak-free elevation (Task 3 BayesQ 0.7952, Task 4 SWA 0.9567, Tasks 1-2 & 5) (#40)
- Merged: `8b5c4a86390e` | Kaggle public: –

### 2026-11-12 – `6192ed08e604` – #34

- Commit: `af4c26f` by Wisit Suwannao
- Prev: `d5fb67040c8f` | Δrows vs prev: 13,445 (51.9%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 8) (#34)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `d5fb67040c8f` – #33

- Commit: `470ce3c` by Wisit Suwannao
- Prev: `6192ed08e604` | Δrows vs prev: 13,445 (51.9%)
- Why: feat: elevate Tasks 1-5 with Bayes-Optimal Dynamic Quantile Hurdle and SOTA Ensembles (Run 8) (#33)
- Merged: `2355d36d5abe` | Kaggle public: 0.42367

### 2026-11-10 – `6192ed08e604` – #25

- Commit: `24ce629` by Wisit Suwannao
- Prev: `d531284b6390` | Δrows vs prev: 13,444 (51.9%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA run 4 elevation targeting >=0.80 benchmark score (#25)
- Merged: `5eca03fe5225` | Kaggle public: 0.67352

### 2026-11-10 – `d531284b6390` – #22

- Commit: `ecca1dc` by Wisit Suwannao
- Prev: `6192ed08e604` | Δrows vs prev: 13,444 (51.9%)
- Why: feat: test submission (Task 3 Hurdle + Task 4 numeric 0-4) (#22)
- Merged: `e0c9e62e2d39` | Kaggle public: 0.51909

### 2026-11-06 – `6192ed08e604` – #13

- Commit: `0650975` by Wisit Suwannao
- Prev: `2bef47200a70` | Δrows vs prev: 13,273 (51.3%)
- Why: feat: elevate all 5 tasks (Task 1-5) beating baseline with SOTA models (#13)
- Merged: `8adba6602333` | Kaggle public: 0.63573

### 2026-11-02 – `2bef47200a70` – #3

- Commit: `ac8f17f` by bbb
- Prev: `323b6e72e54c` | Δrows vs prev: 25,889 (100.0%)
- Why: ci: automate and deduplicate Kaggle submissions (#3)
- Merged: `601c16363890` | Kaggle public: 0.05671

### 2026-11-02 – `323b6e72e54c` – #2

- Commit: `e6226cf` by bbb
- Prev: – | Δrows vs prev: new file
- Why: ci: add submission merge check and kaggle submit workflows (#2)
- Merged: `48a77166cd56` | Kaggle public: 0.12007
