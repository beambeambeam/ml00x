# Task 1 submission (`submit/index.csv`)

> Agents keep this file. Rules: `AGENTS.md` → Submission history. Newest entry first. Never edit or delete old changelog entries.

## Current

- **sha12:** `af57ef293a3a` (2026-11-21, `6324964`, #78)
- **Active Version:** **exp 005**, account cohort likelihoods + exp 004
- **Generating Script:** `task1/run_account_cohorts.py`. Run `uv run python run_account_cohorts.py --promote` from `task1/`.
- **Required Artifacts:** `task1/predictions/experiments/004/{oof,test}.csv` and `task1/models/experiments/004/stack_meta.json`
- **Outputs:** `task1/predictions/experiments/005/{oof,test}.csv`, `task1/models/experiments/005/metrics.json`, and `task1/assets/experiments/005/{results,per-fold-f2}.csv`
- **OOF Metrics:** F2 `0.840965` (+`0.001388`), ROC AUC `0.896463`, PR AUC `0.840479`, log loss `0.351097`. F2 rose in all five folds. Paired row bootstrap 95% interval for the F2 gain is `[+0.000743, +0.002039]`.
- **Validation:** Cohort ranges and blend weight came from exploratory analysis of the labeled data. Fold training estimated cohort weights, skill transforms, and classifier coefficients. Thresholds were tuned on the other OOF folds.
- **Test Distribution (`25,889` rows):** `0`: `10,541` (`40.7%`), `1`: `15,348` (`59.3%`)

## Changelog

### 2026-11-21 – `af57ef293a3a` – #78

- Commit: `6324964` by beambeambeam
- Prev: `4817f4300bce` | Δrows vs prev: 311 (1.2%)
- Why: Experiment 005 combines account cohort likelihoods and a regularized skill classifier with experiment 004. Five-fold OOF F2 increased from `0.839578` to `0.840965`. F2 with incomplete account fields increased from `0.825042` to `0.827285`.
- Merged: `faeb950de6f5` | Kaggle public: –

### 2026-11-20 – `4817f4300bce` – #74

- Commit: `2df195e` by Palapluem (Wisit Suwannao)
- Prev: `fd65d3e13622` | Δrows vs prev: 295 (1.1%)
- Why: feat(task1): exp 004 Pure-Train DGP + Interior CatBoost/LGBM ensemble (`task1/notebooks/experiments/004.ipynb`; `nested_f2 0.839578` vs exp 003 `0.838777`, `obs5_f2 0.860511` vs `0.859353`, `roc_auc 0.894765` vs `0.893780`, `log_loss 0.355336` vs `0.362346`; paired bootstrap win `99.1%` all / `98.7%` `obs==5`; Pure-Train 1D+2D support clamp `d0=7,945` / `d1=3,428`)
- Merged: `b7f0765d8736` | Kaggle public: 0.86362

### 2026-11-20 – `fd65d3e13622` – #68

- Commit: `3b82623` by bbb
- Prev: `394208ee5a6d` | Δrows vs prev: 217 (0.8%)
- Why: feat: Task 1 experiment 003 – 15-feature LightGBM seed/config ensemble (#68)
- Merged: `1e0fec69f3d5` | Kaggle public: 0.59631

### 2026-11-20 – `394208ee5a6d` – #66

- Commit: `52fc1e3` by Wisit Suwannao
- Prev: `427fb07c946e` | Δrows vs prev: 329 (1.3%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 22 Kaggle submission (#66)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `427fb07c946e` – #65

- Commit: `5014990` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 329 (1.3%)
- ↩ Same file as 2026-11-19 `796efb1` (#59)
- Why: feat: Run 22 Task 4 Pure-Train Uniform-Prior Debias(a=0.860) + DINOv2-StageC (#65)
- Merged: `b45d08b2eabc` | Kaggle public: 0.86337

### 2026-11-20 – `394208ee5a6d` – #64

- Commit: `b68ac80` by Wisit Suwannao
- Prev: `427fb07c946e` | Δrows vs prev: 329 (1.3%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 21 Kaggle submission (#64)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `427fb07c946e` – #63

- Commit: `5ff5a3d` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 329 (1.3%)
- ↩ Same file as 2026-11-19 `796efb1` (#59)
- Why: feat: Run 21 Task 4 5-Fold SGKF DINOv2-StageC Nightmare-Hardened (#63)
- Merged: `95d9db7f8078` | Kaggle public: 0.86146

### 2026-11-19 – `394208ee5a6d` – #62

- Commit: `987486f` by Wisit Suwannao
- Prev: `427fb07c946e` | Δrows vs prev: 329 (1.3%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 20 Kaggle submission (#62)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `427fb07c946e` – #61

- Commit: `ca1bd6a` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 329 (1.3%)
- ↩ Same file as 2026-11-19 `796efb1` (#59)
- Why: feat: Run 20 Task 4 pure-train Hard+Iso Nested SGKF + Train-Prior debias (#61)
- Merged: `820b12571d07` | Kaggle public: 0.86153

### 2026-11-19 – `394208ee5a6d` – #60

- Commit: `8f040db` by Wisit Suwannao
- Prev: `427fb07c946e` | Δrows vs prev: 329 (1.3%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 19-C Kaggle submission (#60)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `427fb07c946e` – #59

- Commit: `796efb1` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 329 (1.3%)
- Why: feat: Run 19-C strict zero-test-parameter compliance across Tasks 1 & 4 (#59)
- Merged: `42b91194dc33` | Kaggle public: 0.85811

### 2026-11-19 – `394208ee5a6d` – #58

- Commit: `c231933` by Wisit Suwannao
- Prev: `9b5ada986cea` | Δrows vs prev: 324 (1.3%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 18 Kaggle submission (#58)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `9b5ada986cea` – #57

- Commit: `5acf8b8` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 324 (1.3%)
- Why: feat: Run 18 100% Pure-Train DGP Elevation across Tasks 1-5 (#57)
- Merged: `3f236f5e932e` | Kaggle public: 0.85908

### 2026-11-17 – `394208ee5a6d` – #55

- Commit: `38b3bab` by Wisit Suwannao
- Prev: `4c3add4260b1` | Δrows vs prev: 317 (1.2%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 17 Kaggle submission (#55)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-17 – `4c3add4260b1` – #54

- Commit: `50874df` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 317 (1.2%)
- ↩ Same file as 2026-11-16 `c4b21da` (#50)
- Why: feat(run17): upgrade Task 4 5-Fold SGKF E2E+ViT Stack & Task 5 Gaussian/NaN fix (#54)
- Merged: `005367eb3fed` | Kaggle public: 0.84139

### 2026-11-16 – `394208ee5a6d` – #51

- Commit: `f64ac9a` by Wisit Suwannao
- Prev: `4c3add4260b1` | Δrows vs prev: 317 (1.2%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 16 Kaggle submission (#51)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-16 – `4c3add4260b1` – #50

- Commit: `c4b21da` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 317 (1.2%)
- Why: feat(run16): upgrade Task 3 V3+MPS & Task 5 4-Fam Raw-NaN + Run 15 T1/T2/T4 (#50)
- Merged: `d57456bd72f3` | Kaggle public: 0.83969

### 2026-11-15 – `394208ee5a6d` – #47

- Commit: `c0b808a` by Wisit Suwannao
- Prev: `c05e64cc7b46` | Δrows vs prev: 285 (1.1%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 14 Kaggle submission (#47)
- Merged: `7a843b787104` | Kaggle public: 0.59607

### 2026-11-15 – `c05e64cc7b46` – #46

- Commit: `c7dc931` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 285 (1.1%)
- Why: feat(run14): upgrade Tasks 2 & 3 zero-leakage nested ensembles + Task 4 4xViT+SWA stack (#46)
- Merged: `1362e3d02daa` | Kaggle public: 0.83904

### 2026-11-14 – `394208ee5a6d` – #45

- Commit: `4cb35e5` by Wisit Suwannao
- Prev: `ce934f31d1de` | Δrows vs prev: 242 (0.9%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Kaggle submission (#45)
- Merged: `ea7b3897e3cd` | Kaggle public: –

### 2026-11-14 – `ce934f31d1de` – #44

- Commit: `997bcd2` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 242 (0.9%)
- Why: feat(run12): upgrade all 5 tasks + Task 3 dual-grid Bayes quantile hurdle (#44)
- Merged: `1727f0c63c80` | Kaggle public: 0.81773

### 2026-11-14 – `394208ee5a6d` – #43

- Commit: `47239c2` by Wisit Suwannao
- Prev: `2d344b704405` | Δrows vs prev: 381 (1.5%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11 retry) (#43)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-14 – `2d344b704405` – #42

- Commit: `d4f7793` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 381 (1.5%)
- ↩ Same file as 2026-11-13 `9bc7502` (#40)
- Why: feat: Run 11 SOTA leak-free elevation (retry post daily reset) (#42)
- Merged: `8b5c4a86390e` | Kaggle public: 0.81491

### 2026-11-13 – `394208ee5a6d` – #41

- Commit: `411aae1` by Wisit Suwannao
- Prev: `2d344b704405` | Δrows vs prev: 381 (1.5%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11) (#41)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-13 – `2d344b704405` – #40

- Commit: `9bc7502` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 381 (1.5%)
- Why: feat: Run 11 SOTA leak-free elevation (Task 3 BayesQ 0.7952, Task 4 SWA 0.9567, Tasks 1-2 & 5) (#40)
- Merged: `8b5c4a86390e` | Kaggle public: –

### 2026-11-13 – `394208ee5a6d` – #38

- Commit: `a1bee4c` by Wisit Suwannao
- Prev: `19ed96da1697` | Δrows vs prev: 632 (2.4%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 10) (#38)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-13 – `19ed96da1697` – #37

- Commit: `bc06d1f` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 632 (2.4%)
- Why: feat: Run 10 SOTA elevation with 2x2 Quadrant GBDT (Task 2: 0.7524) and 5-Paradigm Deep Autoencoder (Task 5: F3 0.90+) (#37)
- Merged: `3e418d058152` | Kaggle public: 0.72682

### 2026-11-12 – `394208ee5a6d` – #32

- Commit: `a23848d` by Wisit Suwannao
- Prev: `9d8b2e455c6f` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (#32) (#32)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `9d8b2e455c6f` – #31

- Commit: `a78540d` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA Run 7 elevation (Task 5 PyOD PCA 0.05 contamination + Task 2 Cohort 0.966 + Task 4 DINOv2) (#31)
- Merged: `bc8836f51f7d` | Kaggle public: 0.67549

### 2026-11-12 – `394208ee5a6d` – #30

- Commit: `f00d286` by Wisit Suwannao
- Prev: `9d8b2e455c6f` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (#30) (#30)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `9d8b2e455c6f` – #29

- Commit: `2b8b419` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA Run 6 champion elevation (Task 4 DINOv2 0.947, Task 2 cohort 0.966, Task 5 F3 tripwires) (#29)
- Merged: `ba9f3a6f82b1` | Kaggle public: 0.71761

### 2026-11-10 – `394208ee5a6d` – #26

- Commit: `f19961b` by Wisit Suwannao
- Prev: `9d8b2e455c6f` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (#26)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-10 – `9d8b2e455c6f` – #25

- Commit: `24ce629` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA run 4 elevation targeting >=0.80 benchmark score (#25)
- Merged: `5eca03fe5225` | Kaggle public: 0.67352

### 2026-11-07 – `394208ee5a6d` – #18

- Commit: `82b824f` by Wisit Suwannao
- Prev: `9d8b2e455c6f` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (#18)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `9d8b2e455c6f` – #17

- Commit: `bdb7de6` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA run 3 with calibrated DINOv2 and high-confidence anomaly tripwires (#17)
- Merged: `e80019f8da94` | Kaggle public: 0.63573

### 2026-11-07 – `394208ee5a6d` – #16

- Commit: `2d82be0` by Wisit Suwannao
- Prev: `9d8b2e455c6f` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (#16)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `9d8b2e455c6f` – #15

- Commit: `e89f219` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: elevate all 5 tasks targeting >=0.80 score with DINOv2 and F3 calibration (#15)
- Merged: `d36e163f2f81` | Kaggle public: 0.63286

### 2026-11-07 – `394208ee5a6d` – #14

- Commit: `d96af9d` by Wisit Suwannao
- Prev: `9d8b2e455c6f` | Δrows vs prev: 190 (0.7%)
- ↩ Same file as 2026-11-03 `b8e1643` (#6)
- Why: chore: restore teammate tasks to upstream state post-submission (#14)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-06 – `9d8b2e455c6f` – #13

- Commit: `0650975` by Wisit Suwannao
- Prev: `394208ee5a6d` | Δrows vs prev: 190 (0.7%)
- Why: feat: elevate all 5 tasks (Task 1-5) beating baseline with SOTA models (#13)
- Merged: `8adba6602333` | Kaggle public: 0.63573

### 2026-11-03 – `394208ee5a6d` – #6

- Commit: `b8e1643` by bbb
- Prev: `031f525593c6` | Δrows vs prev: 471 (1.8%)
- Why: feat(task1): experiment 002 – fine-split GBDTs, test-like validation, and blend (#6)
- Merged: `b6a38ac93a15` | Kaggle public: 0.17137

### 2026-11-02 – `031f525593c6` – #4

- Commit: `5545b44` by bbb
- Prev: `1beb2da6d0b7` | Δrows vs prev: 15,612 (60.3%)
- Why: feat(task1): EDA, baseline GBDT experiment, and first submission (#4)
- Merged: `d1eff4ff6565` | Kaggle public: 0.17183

### 2026-11-02 – `1beb2da6d0b7` – #2

- Commit: `e6226cf` by bbb
- Prev: – | Δrows vs prev: new file
- Why: ci: add submission merge check and kaggle submit workflows (#2)
- Merged: `48a77166cd56` | Kaggle public: 0.12007
