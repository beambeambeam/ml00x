# Submission history

> Agents keep this file. Rules: `AGENTS.md` → Submission history. Newest entry first. Never edit or delete old changelog entries.

Get exact `sha12` values and Δrows from `uv run submission/merge.py --baseline <main checkout>`.
The merged `sha12` equals the `[sha:<hash>]` tag in each Kaggle note.

## Current

| Task | sha12 | Since | Source |
| --- | --- | --- | --- |
| [Task 1](task1/submit/submit.md) | `af57ef293a3a` | 2026-11-21, #78 | exp 005 – `task1/run_account_cohorts.py` |
| [Task 2](task2/submit/submit.md) | `34d927a40901` | 2026-11-21, #79 | Run 22 models, affine export fix – `task2/run_bayes_export.py` |
| [Task 3](task3/submit/submit.md) | `3646bdd6dda1` | 2026-11-19, #57 | Run 22 – `task3/task3_monthly_spending.ipynb` |
| [Task 4](task4/submit/submit.md) | `1cb773dd067e` | 2026-11-21, #77 | Run 22 – `task4/task4_game_title_detection.ipynb` (reverted from #76) |
| [Task 5](task5/submit/submit.md) | `5e57282c6459` | 2026-11-20, #71 | Archetype rules – `task5/archetype_rules.py` |

- **Merged sha12:** `904a1fd0976c` – Kaggle public: –
- **Best Kaggle public so far:** 0.86363 (merged `b3ea360f52c3`, commit `33160ae`)

## Merged history

One row per `main` commit that changed any `task{N}/submit/index.csv`. "↩" marks a merged file that equals an earlier one. Kaggle public lists each score for that merged file.

| Date | Commit | PR | Merged sha12 | Tasks changed | Kaggle public | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-21 | `2f988af` | #79 | `904a1fd0976c` | 2 | – | Task 2 affine export fix; frozen-OOF macro F1 `0.753158` → `0.759316`, all five folds improve; 1,622 rows changed |
| 2026-11-21 | `6324964` | #78 | `faeb950de6f5` | 1 | – | Task 1 exp 005 account cohorts and skill classifier, 311 rows changed and OOF F2 `0.840965` |
| 2026-11-21 | `pending` | #77 | `b7f0765d8736` | 4 | 0.86362 | ↩ same file as `2df195e`. fix(task4): revert task4 submit/index.csv to Run 22 after Run 23 0.86310 regression (#77) |
| 2026-11-21 | `bbec716` | #76 | `6caa214ed301` | 4 | 0.86310 | feat(task4): Run 23 add 5-fold SGKF Nightmare-Hardened ConvNeXtV2-StageC (#76) |
| 2026-11-21 | `2df195e` | #74 | `b7f0765d8736` | 1 | 0.86362 | feat(task1): exp 004 Pure-Train DGP + Interior CatBoost/LGBM ensemble (#74) |
| 2026-11-20 | `05a762a` | #71 | `1f26381be9b0` | 5 | – | feat: Task 5 archetype rules with calibrated NaN-unstable recovery (#71) |
| 2026-11-20 | `33160ae` | #70 | `b3ea360f52c3` | 5 | 0.86363 | fix: restore Run 22 task5 submit/index.csv (#70) |
| 2026-11-20 | `8654961` | #69 | `296e7a3ae27a` | 2, 4 | 0.84113 | fix: restore Run 22 task2 and task4 submit/index.csv (#69) |
| 2026-11-20 | `3b82623` | #68 | `1e0fec69f3d5` | 1 | 0.59631 | feat: Task 1 experiment 003 – 15-feature LightGBM seed/config ensemble (#68) |
| 2026-11-20 | `52fc1e3` | #66 | `f78cc560de14` | 1, 2, 4, 5 | – | ↩ same file as `c231933`. chore: restore teammate task1,2,4,5 submit/index.csv after Run 22 Kaggle submission (#66) |
| 2026-11-20 | `5014990` | #65 | `b45d08b2eabc` | 1, 2, 4, 5 | 0.86337 | feat: Run 22 Task 4 Pure-Train Uniform-Prior Debias(a=0.860) + DINOv2-StageC (#65) |
| 2026-11-20 | `b68ac80` | #64 | `f78cc560de14` | 1, 2, 4, 5 | – | ↩ same file as `c231933`. chore: restore teammate task1,2,4,5 submit/index.csv after Run 21 Kaggle submission (#64) |
| 2026-11-20 | `5ff5a3d` | #63 | `95d9db7f8078` | 1, 2, 4, 5 | 0.86146 | feat: Run 21 Task 4 5-Fold SGKF DINOv2-StageC Nightmare-Hardened (#63) |
| 2026-11-19 | `987486f` | #62 | `f78cc560de14` | 1, 2, 4, 5 | – | ↩ same file as `c231933`. chore: restore teammate task1,2,4,5 submit/index.csv after Run 20 Kaggle submission (#62) |
| 2026-11-19 | `ca1bd6a` | #61 | `820b12571d07` | 1, 2, 4, 5 | 0.86153 | feat: Run 20 Task 4 pure-train Hard+Iso Nested SGKF + Train-Prior debias (#61) |
| 2026-11-19 | `8f040db` | #60 | `f78cc560de14` | 1, 2, 4, 5 | – | ↩ same file as `c231933`. chore: restore teammate task1,2,4,5 submit/index.csv after Run 19-C Kaggle submission (#60) |
| 2026-11-19 | `796efb1` | #59 | `42b91194dc33` | 1, 2, 4, 5 | 0.85811 | feat: Run 19-C strict zero-test-parameter compliance across Tasks 1 & 4 (#59) |
| 2026-11-19 | `c231933` | #58 | `f78cc560de14` | 1, 2, 4, 5 | – | chore: restore teammate task1,2,4,5 submit/index.csv after Run 18 Kaggle submission (#58) |
| 2026-11-19 | `5acf8b8` | #57 | `3f236f5e932e` | 1, 2, 3, 4, 5 | 0.85908 | feat: Run 18 100% Pure-Train DGP Elevation across Tasks 1-5 (#57) |
| 2026-11-17 | `38b3bab` | #55 | `ac42bd0a2e56` | 1, 2, 4, 5 | – | ↩ same file as `f64ac9a`. chore: restore teammate tasks (task1, task2, task4, task5) after Run 17 Kaggle submission (#55) |
| 2026-11-17 | `50874df` | #54 | `005367eb3fed` | 1, 2, 4, 5 | 0.84139 | feat(run17): upgrade Task 4 5-Fold SGKF E2E+ViT Stack & Task 5 Gaussian/NaN fix (#54) |
| 2026-11-16 | `f64ac9a` | #51 | `ac42bd0a2e56` | 1, 2, 4, 5 | – | chore: restore teammate tasks (task1, task2, task4, task5) after Run 16 Kaggle submission (#51) |
| 2026-11-16 | `c4b21da` | #50 | `d57456bd72f3` | 1, 2, 3, 4, 5 | 0.83969 | feat(run16): upgrade Task 3 V3+MPS & Task 5 4-Fam Raw-NaN + Run 15 T1/T2/T4 (#50) |
| 2026-11-15 | `c0b808a` | #47 | `7a843b787104` | 1, 2, 4, 5 | 0.59607 | chore: restore teammate tasks (task1, task2, task4, task5) after Run 14 Kaggle submission (#47) |
| 2026-11-15 | `c7dc931` | #46 | `1362e3d02daa` | 1, 2, 3, 4, 5 | 0.83904 | feat(run14): upgrade Tasks 2 & 3 zero-leakage nested ensembles + Task 4 4xViT+SWA stack (#46) |
| 2026-11-14 | `4cb35e5` | #45 | `ea7b3897e3cd` | 1, 2, 4, 5 | – | chore: restore teammate tasks (task1, task2, task4, task5) after Kaggle submission (#45) |
| 2026-11-14 | `997bcd2` | #44 | `1727f0c63c80` | 1, 2, 3, 4, 5 | 0.81773 | feat(run12): upgrade all 5 tasks + Task 3 dual-grid Bayes quantile hurdle (#44) |
| 2026-11-14 | `47239c2` | #43 | `c88d74f878e6` | 1, 2, 4, 5 | – | ↩ same file as `411aae1`. chore: restore teammate tasks to upstream state post-submission (Run 11 retry) (#43) |
| 2026-11-14 | `d4f7793` | #42 | `8b5c4a86390e` | 1, 2, 4, 5 | 0.81491 | ↩ same file as `9bc7502`. feat: Run 11 SOTA leak-free elevation (retry post daily reset) (#42) |
| 2026-11-13 | `411aae1` | #41 | `c88d74f878e6` | 1, 2, 4, 5 | – | chore: restore teammate tasks to upstream state post-submission (Run 11) (#41) |
| 2026-11-13 | `9bc7502` | #40 | `8b5c4a86390e` | 1, 2, 3, 4, 5 | – | feat: Run 11 SOTA leak-free elevation (Task 3 BayesQ 0.7952, Task 4 SWA 0.9567, Tasks 1-2 & 5) (#40) |
| 2026-11-13 | `2602a80` | #39 | `35eeed2758fd` | 5 | 0.59580 | feat:  task5 add account security anomaly detection pipeline (#39) |
| 2026-11-13 | `a1bee4c` | #38 | `06683c0cfd15` | 1, 2, 4, 5 | – | ↩ same file as `f19961b`. chore: restore teammate tasks to upstream state post-submission (Run 10) (#38) |
| 2026-11-13 | `bc06d1f` | #37 | `3e418d058152` | 1, 2, 4, 5 | 0.72682 | feat: Run 10 SOTA elevation with 2x2 Quadrant GBDT (Task 2: 0.7524) and 5-Paradigm Deep Autoencoder (Task 5: F3 0.90+) (#37) |
| 2026-11-12 | `485cf48` | #36 | `06683c0cfd15` | 2, 4, 5 | – | ↩ same file as `f19961b`. chore: restore teammate tasks to upstream state post-submission (Run 9) (#36) |
| 2026-11-12 | `8fd32a7` | #35 | `f71ce3e2349b` | 2, 4, 5 | 0.71770 | feat: Run 9 SOTA elevation with new TA dataset (Task 2 Cohort GBDT 0.7508, Task 4 ConvNeXt-Tiny MLP 0.9019, Task 5 F3 Anomaly) (#35) |
| 2026-11-12 | `af4c26f` | #34 | `06683c0cfd15` | 3, 4, 5 | – | ↩ same file as `f19961b`. chore: restore teammate tasks to upstream state post-submission (Run 8) (#34) |
| 2026-11-12 | `470ce3c` | #33 | `2355d36d5abe` | 3, 4, 5 | 0.42367 | feat: elevate Tasks 1-5 with Bayes-Optimal Dynamic Quantile Hurdle and SOTA Ensembles (Run 8) (#33) |
| 2026-11-12 | `a23848d` | #32 | `06683c0cfd15` | 1, 2, 4, 5 | – | ↩ same file as `f19961b`. chore: restore teammate tasks to upstream state post-submission (#32) (#32) |
| 2026-11-12 | `a78540d` | #31 | `bc8836f51f7d` | 1, 2, 4, 5 | 0.67549 | feat: SOTA Run 7 elevation (Task 5 PyOD PCA 0.05 contamination + Task 2 Cohort 0.966 + Task 4 DINOv2) (#31) |
| 2026-11-12 | `f00d286` | #30 | `06683c0cfd15` | 1, 2, 4, 5 | – | ↩ same file as `f19961b`. chore: restore teammate tasks to upstream state post-submission (#30) (#30) |
| 2026-11-12 | `2b8b419` | #29 | `ba9f3a6f82b1` | 1, 2, 4, 5 | 0.71761 | feat: SOTA Run 6 champion elevation (Task 4 DINOv2 0.947, Task 2 cohort 0.966, Task 5 F3 tripwires) (#29) |
| 2026-11-12 | `27a8a04` | #28 | `06683c0cfd15` | 2, 4, 5 | – | ↩ same file as `f19961b`. chore: restore teammate tasks to upstream state post-submission (#28) |
| 2026-11-12 | `456a7d5` | #27 | `aaa52adb72f8` | 2, 4, 5 | 0.65531 | feat: SOTA Run 5 elevation (Task 2 cohort GBDT 0.966, Task 4 DINOv2 0.916, Task 5 F3 ensemble) (#27) |
| 2026-11-10 | `f19961b` | #26 | `06683c0cfd15` | 1, 2, 4, 5 | – | chore: restore teammate tasks to upstream state post-submission (#26) |
| 2026-11-10 | `24ce629` | #25 | `5eca03fe5225` | 1, 2, 3, 4, 5 | 0.67352 | feat: SOTA run 4 elevation targeting >=0.80 benchmark score (#25) |
| 2026-11-10 | `8d183dc` | #23 | `a1623e0ddc82` | 4 | – | chore: restore teammate tasks to upstream state post-submission (#23) |
| 2026-11-10 | `ecca1dc` | #22 | `e0c9e62e2d39` | 3, 4 | 0.51909 | feat: test submission (Task 3 Hurdle + Task 4 numeric 0-4) (#22) |
| 2026-11-10 | `ff6a94f` | #21 | `e7ee94c8314e` | 2 | – | fix(task2): submit RealMLP-TD predictions as new best index.csv (#21) |
| 2026-11-09 | `f34c92b` | #19 | `5dc3d5a113e8` | 2 | 0.48054 | fix(task2): restore top-pick submission predictions (public 0.58091) (#19) |
| 2026-11-07 | `82b824f` | #18 | `1482aa113562` | 1, 2, 4, 5 | – | ↩ same file as `d96af9d`. chore: restore teammate tasks to upstream state post-submission (#18) |
| 2026-11-07 | `bdb7de6` | #17 | `e80019f8da94` | 1, 2, 4, 5 | 0.63573 | feat: SOTA run 3 with calibrated DINOv2 and high-confidence anomaly tripwires (#17) |
| 2026-11-07 | `2d82be0` | #16 | `1482aa113562` | 1, 2, 4, 5 | – | ↩ same file as `d96af9d`. chore: restore teammate tasks to upstream state post-submission (#16) |
| 2026-11-07 | `e89f219` | #15 | `d36e163f2f81` | 1, 2, 4, 5 | 0.63286 | feat: elevate all 5 tasks targeting >=0.80 score with DINOv2 and F3 calibration (#15) |
| 2026-11-07 | `d96af9d` | #14 | `1482aa113562` | 1, 2, 4, 5 | – | chore: restore teammate tasks to upstream state post-submission (#14) |
| 2026-11-06 | `0650975` | #13 | `8adba6602333` | 1, 2, 3, 4, 5 | 0.63573 | feat: elevate all 5 tasks (Task 1-5) beating baseline with SOTA models (#13) |
| 2026-11-05 | `b6c5d61` | #10 | `881d6d9c1cff` | 4 | 0.47978 | feat(task4): add game title image classification pipeline (#10) |
| 2026-11-03 | `a286632` | #11 | `b8853c7b0557` | 2 | 0.47978 | feat(task2): add segment research artifacts and top model (#11) |
| 2026-11-03 | `f552106` | #9 | `b6a38ac93a15` | 2, 4, 5 | – | ↩ same file as `b8e1643`. chore(tasks-2-4-5): reset secondary tasks to scaffolding for team contributors (#9) |
| 2026-11-03 | `ba483f9` | #7 | `4891205ca6cc` | 2, 4, 5 | 0.58091 | feat(tasks-2-4-5): elevate secondary tasks with trained ensemble models and test predictions (#7) |
| 2026-11-03 | `b8e1643` | #6 | `b6a38ac93a15` | 1 | 0.17137 | feat(task1): experiment 002 – fine-split GBDTs, test-like validation, and blend (#6) |
| 2026-11-02 | `5545b44` | #4 | `d1eff4ff6565` | 1 | 0.17183 | feat(task1): EDA, baseline GBDT experiment, and first submission (#4) |
| 2026-11-02 | `ac8f17f` | #3 | `601c16363890` | 3 | 0.05671 | ci: automate and deduplicate Kaggle submissions (#3) |
| 2026-11-02 | `e6226cf` | #2 | `48a77166cd56` | 1, 2, 3, 4, 5 | 0.12007 | ci: add submission merge check and kaggle submit workflows (#2) |

## Workflow (`main`-only + `submit.md`)

1. **No separate throwaway branches / restore PRs:** Develop and land verified updates directly on `main`.
2. **Always keep code + model bundles on `main`:** Any `task{N}/submit/index.csv` update must include the corresponding notebook/script and artifacts in `main`, and update both root `submit.md` and `task{N}/submit/submit.md` in the same commit (rules: `AGENTS.md` → Submission history).
3. **If a Kaggle submission regresses ("revert only `submit/index.csv`"):** Keep the experiment code/notebooks on `main` for reference, and revert **only** `task{N}/submit/index.csv` + update `submit.md` and `task{N}/submit/submit.md`:
   ```bash
   git checkout <good-commit-sha> -- task<N>/submit/index.csv task<N>/submit/submit.md submit.md
   uv run submission/merge.py
   ```
4. **Submission check before push:**
   ```bash
   uv run submission/merge.py
   ```
   *(Kaggle submissions remain manual via GitHub Actions -> `Submit to Kaggle` -> `Run workflow` on `main`.)*
