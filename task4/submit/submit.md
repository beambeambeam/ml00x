# Task 4 submission (`submit/index.csv`)

> Agents keep this file. Rules: `AGENTS.md` → Submission history. Newest entry first. Never edit or delete old changelog entries.

## Current

- **sha12:** `f9b481b506b7` (since 2026-11-23, #87)
- **Active Version:** **Run 25-V2 — Trio-Consensus Shift Gate Dual-Regime Fusion (`CNXv3-Base + DINOv3L-Hard + EVA02L-Hard + DINOv2 + Base ViTs`)** (`Palapluem`, PR `#87`)
- **Generating Script / Notebook:** `task4/scripts/fuse_t4_sota_trio.py`, `task4/task4_game_title_detection.ipynb`
- **Required Artifacts:**
  - `task4/predictions/model-level/task4-sgkf-e2e-bundle.npz` (5-fold SGKF Pure-Train E2E `ConvNeXtV2`, `DINOv2-StageA/C`, `MLP-V3`)
  - `task4/predictions/model-level/task4-cnxv3-sgkf-hard-bundle.npz` & `task4/predictions/model-level/task4-cnxv3-{orig2,hard6}-{oof,test-predictions}.csv`
  - `task4/predictions/model-level/task4-dinov3l-sgkf-hard-bundle.npz` & `task4/predictions/model-level/task4-dinov3l-{orig2,hard6}-{oof,test-predictions}.csv`
  - `task4/predictions/model-level/task4-eva02l-sgkf-hard-bundle.npz` & `task4/predictions/model-level/task4-eva02l-{orig2,hard6}-{oof,test-predictions}.csv`
  - 8 base ViT CSVs in `task4/predictions/model-level/`: `task4-{dinov3-vitb,dinov3-vitl,eva02-base,eva02-large}-{oof,test-predictions}.csv`
  - `task4/models/t4_sota_fusion_meta.json`, `task4/data/train.csv`, `task4/data/test.csv`
- **Method & OOF Metrics (`31,546` train images, 100% Pure-Train):**
  - Dual-Regime Temperature-Calibrated Log-Odds SLSQP Clean Pool (`EVA02-Large=0.1960`, `EVA02L-Hard-Orig2=0.1906`, `DINOv3L-Hard-Orig2=0.1822`, `CNXv3-Base-Hard6=0.1435`, `EVA02L-Hard-Hard6=0.0853`, `DINOv3-ViT-L=0.0631`, `CNXv3-Base-Orig2=0.0566`, `DINOv3L-Hard-Hard6=0.0460`, `EVA02-Base=0.0354`, `ConvNeXtV2-SGKF=0.0008`, `MLP-V3-SGKF=0.0005`) + ExtHard6 Sub-Pool (`CNXv2-StageC-6TTA=0.3230`, `DINOv3L-Hard-Hard6=0.2896`, `EVA02L-Hard-Hard6=0.2629`, `CNXv3-Base-Hard6=0.0695`, `DINOv2-StageC-6TTA=0.0549`) + Pure-Train 3-Family Consensus Restoration Shift Gate (`tau_js_98pct_iso=0.006393`, `gamma=0.1`, `test_rows_gated=770`) + Pure-Train Nested Prior Debias (`alpha*=0.6`, `[0.6, 0.6, 0.6, 0.6, 0.6]`)
  - `OOF Macro-F1 = 0.994412` | `OOF LogLoss = 0.019466` | `Iso(<0.75) F1 = 0.995665` | `Corrupt F1 = 0.989838` | `ExtHard6 F1 = 0.970477` | `ExtHard6 LogLoss = 0.087769` | `Paired Bootstrap vs Run 22 (0.991174) = 1.0000`
- **Test Distribution (`25,889` rows):** `0` (`7-11k`): `3,707` (`14.3%`), `1` (`fiveN`): `6,729` (`26.0%`), `2` (`freefried`): `3,955` (`15.3%`), `3` (`pube`): `6,334` (`24.5%`), `4` (`roblock`): `5,164` (`19.9%`)
  - *Note:* `index.csv` stores canonical numeric labels `0..4` (`AGENTS.md` schema).

## Changelog

### 2026-11-23 – `f9b481b506b7` – #87

- Commit: `fa90a83` by Palapluem (Wisit Suwannao)
- Prev: `e146d960e1a0` | Δrows vs prev: 41 (0.2%)
- Why: feat(task4): Run 25-V2 Pure-Train 3-Family Consensus Restoration Shift Gate (`tau_js_98pct_iso=0.006393`, `gamma=0.10` across all 5 outer folds, `test_rows_gated=770`, `alpha*=0.60`), lifting `all_oof_f1 0.994122 -> 0.994412` (`+0.000290`, `500/500` bootstrap vs Run 25), `iso_oof_f1 0.995278 -> 0.995665`, `corrupt_oof_f1 0.989391 -> 0.989838`, plus `siglip_l16` auto-discovery
- Merged: `e3b50d7d48a7` | Kaggle public: 0.86545

### 2026-11-23 – `e146d960e1a0` – #84

- Commit: `2024555` by Palapluem (Wisit Suwannao)
- Prev: `cd9689ef4951` | Δrows vs prev: 371 (1.4%)
- Why: feat(task4): Run 25 Dual-Regime SOTA Trio Fusion integrating 5-fold SGKF Nightmare-Hardened EVA02-Large-448 (`EVA02L-Hard`, `OOF_W6 F1 0.991472`, `ExtHard6 F1 0.963912`, `LL 0.048044`) alongside `CNXv3-Base` and `DINOv3L-Hard` (`all_oof_f1 0.994122` vs Run 24 `0.993723`, `iso_oof_f1 0.995278`, `corrupt_oof_f1 0.989391`, `exthard6_oof_f1 0.970477`, `oof_logloss 0.019381`, `paired_bootstrap_vs_r22 1.0000`)
- Merged: `ecadd77e92d2` | Kaggle public: 0.86540

### 2026-11-22 – `cd9689ef4951` – #80

- Commit: `fb0da33` by Palapluem (Wisit Suwannao)
- Prev: `1cb773dd067e` | Δrows vs prev: 793 (3.1%)
- Why: feat(task4): Run 24 Dual-Regime SOTA Fusion integrating 5-fold SGKF Nightmare-Hardened ConvNeXtV3-Base (`CNXv3-Base`) and DINOv3-ViT-L-16 (`DINOv3L-Hard`, `OOF_W6 F1 0.991660`, `ExtHard6 F1 0.962927`) with DINOv2-StageC and Base ViTs (`all_oof_f1 0.993723` vs Run 22 `0.991174`, `iso_oof_f1 0.993871`, `corrupt_oof_f1 0.988965`, `exthard6_oof_f1 0.968777`, `oof_logloss 0.020969`, `paired_bootstrap_vs_r22 1.0000`)
- Merged: `b2f908917ce1` | Kaggle public: 0.86536

### 2026-11-21 – `1cb773dd067e` – #77

- Commit: `pending` by Palapluem (Wisit Suwannao)
- Prev: `7574dd3c1c40` | Δrows vs prev: 240 (0.9%)
- ↩ Same file as 2026-11-20 `5014990` (#65)
- Why: fix(task4): revert task4 submit/index.csv to Run 22 (`1cb773dd067e`) after Run 23 (`0.86310`) regressed `-0.00052` vs `0.86362` (keep Run 23 code/bundle on `main` for reference)
- Merged: `b7f0765d8736` | Kaggle public: 0.86362

### 2026-11-21 – `7574dd3c1c40` – #76

- Commit: `bbec716` by Palapluem (Wisit Suwannao)
- Prev: `1cb773dd067e` | Δrows vs prev: 240 (0.9%)
- Why: feat(task4): Run 23 integrate 5-fold SGKF Nightmare-Hardened ConvNeXtV2-Tiny Stage-C (`CNXv2-StageC`: `All 0.982770`, `ExtHard6 0.945645`, `LL 0.065822`; Run 23 11-stream pool `OOF Macro-F1 0.991577` vs Run 22 `0.991174`, `LL 0.025643` vs `0.026605` [`100%` bootstrap], `Iso F1 0.992266`, `Corrupt F1 0.986601`, `UnifW F1 0.992338` [`88.6%` bootstrap])
- Merged: `6caa214ed301` | Kaggle public: 0.86310

### 2026-11-20 – `1cb773dd067e` – #69

- Commit: `8654961` by bbb
- Prev: `541099d9e792` | Δrows vs prev: 20,573 (79.5%)
- ↩ Same file as 2026-11-20 `5014990` (#65)
- Why: fix: restore Run 22 task2 and task4 submit/index.csv (#69)
- Merged: `296e7a3ae27a` | Kaggle public: 0.84113

### 2026-11-20 – `541099d9e792` – #66

- Commit: `52fc1e3` by Wisit Suwannao
- Prev: `1cb773dd067e` | Δrows vs prev: 20,573 (79.5%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 22 Kaggle submission (#66)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `1cb773dd067e` – #65

- Commit: `5014990` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,573 (79.5%)
- Why: feat: Run 22 Task 4 Pure-Train Uniform-Prior Debias(a=0.860) + DINOv2-StageC (#65)
- Merged: `b45d08b2eabc` | Kaggle public: 0.86337

### 2026-11-20 – `541099d9e792` – #64

- Commit: `b68ac80` by Wisit Suwannao
- Prev: `2c33e69b5351` | Δrows vs prev: 20,579 (79.5%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 21 Kaggle submission (#64)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `2c33e69b5351` – #63

- Commit: `5ff5a3d` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,579 (79.5%)
- Why: feat: Run 21 Task 4 5-Fold SGKF DINOv2-StageC Nightmare-Hardened (#63)
- Merged: `95d9db7f8078` | Kaggle public: 0.86146

### 2026-11-19 – `541099d9e792` – #62

- Commit: `987486f` by Wisit Suwannao
- Prev: `faf2808fe5c6` | Δrows vs prev: 20,576 (79.5%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 20 Kaggle submission (#62)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `faf2808fe5c6` – #61

- Commit: `ca1bd6a` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,576 (79.5%)
- Why: feat: Run 20 Task 4 pure-train Hard+Iso Nested SGKF + Train-Prior debias (#61)
- Merged: `820b12571d07` | Kaggle public: 0.86153

### 2026-11-19 – `541099d9e792` – #60

- Commit: `8f040db` by Wisit Suwannao
- Prev: `4a6e99aa0740` | Δrows vs prev: 20,576 (79.5%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 19-C Kaggle submission (#60)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `4a6e99aa0740` – #59

- Commit: `796efb1` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,576 (79.5%)
- Why: feat: Run 19-C strict zero-test-parameter compliance across Tasks 1 & 4 (#59)
- Merged: `42b91194dc33` | Kaggle public: 0.85811

### 2026-11-19 – `541099d9e792` – #58

- Commit: `c231933` by Wisit Suwannao
- Prev: `b37095451618` | Δrows vs prev: 20,569 (79.5%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 18 Kaggle submission (#58)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `b37095451618` – #57

- Commit: `5acf8b8` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,569 (79.5%)
- Why: feat: Run 18 100% Pure-Train DGP Elevation across Tasks 1-5 (#57)
- Merged: `3f236f5e932e` | Kaggle public: 0.85908

### 2026-11-17 – `541099d9e792` – #55

- Commit: `38b3bab` by Wisit Suwannao
- Prev: `0e0417713b0a` | Δrows vs prev: 20,597 (79.6%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 17 Kaggle submission (#55)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-17 – `0e0417713b0a` – #54

- Commit: `50874df` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,597 (79.6%)
- Why: feat(run17): upgrade Task 4 5-Fold SGKF E2E+ViT Stack & Task 5 Gaussian/NaN fix (#54)
- Merged: `005367eb3fed` | Kaggle public: 0.84139

### 2026-11-16 – `541099d9e792` – #51

- Commit: `f64ac9a` by Wisit Suwannao
- Prev: `575d53194bbf` | Δrows vs prev: 20,558 (79.4%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 16 Kaggle submission (#51)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-16 – `575d53194bbf` – #50

- Commit: `c4b21da` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,558 (79.4%)
- ↩ Same file as 2026-11-15 `c7dc931` (#46)
- Why: feat(run16): upgrade Task 3 V3+MPS & Task 5 4-Fam Raw-NaN + Run 15 T1/T2/T4 (#50)
- Merged: `d57456bd72f3` | Kaggle public: 0.83969

### 2026-11-15 – `541099d9e792` – #47

- Commit: `c0b808a` by Wisit Suwannao
- Prev: `575d53194bbf` | Δrows vs prev: 20,558 (79.4%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 14 Kaggle submission (#47)
- Merged: `7a843b787104` | Kaggle public: 0.59607

### 2026-11-15 – `575d53194bbf` – #46

- Commit: `c7dc931` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,558 (79.4%)
- Why: feat(run14): upgrade Tasks 2 & 3 zero-leakage nested ensembles + Task 4 4xViT+SWA stack (#46)
- Merged: `1362e3d02daa` | Kaggle public: 0.83904

### 2026-11-14 – `541099d9e792` – #45

- Commit: `4cb35e5` by Wisit Suwannao
- Prev: `0c3533c7f0e0` | Δrows vs prev: 20,583 (79.5%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Kaggle submission (#45)
- Merged: `ea7b3897e3cd` | Kaggle public: –

### 2026-11-14 – `0c3533c7f0e0` – #44

- Commit: `997bcd2` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,583 (79.5%)
- Why: feat(run12): upgrade all 5 tasks + Task 3 dual-grid Bayes quantile hurdle (#44)
- Merged: `1727f0c63c80` | Kaggle public: 0.81773

### 2026-11-14 – `541099d9e792` – #43

- Commit: `47239c2` by Wisit Suwannao
- Prev: `58c9a3c015de` | Δrows vs prev: 20,606 (79.6%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11 retry) (#43)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-14 – `58c9a3c015de` – #42

- Commit: `d4f7793` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,606 (79.6%)
- ↩ Same file as 2026-11-13 `9bc7502` (#40)
- Why: feat: Run 11 SOTA leak-free elevation (retry post daily reset) (#42)
- Merged: `8b5c4a86390e` | Kaggle public: 0.81491

### 2026-11-13 – `541099d9e792` – #41

- Commit: `411aae1` by Wisit Suwannao
- Prev: `58c9a3c015de` | Δrows vs prev: 20,606 (79.6%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11) (#41)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-13 – `58c9a3c015de` – #40

- Commit: `9bc7502` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,606 (79.6%)
- Why: feat: Run 11 SOTA leak-free elevation (Task 3 BayesQ 0.7952, Task 4 SWA 0.9567, Tasks 1-2 & 5) (#40)
- Merged: `8b5c4a86390e` | Kaggle public: –

### 2026-11-13 – `541099d9e792` – #38

- Commit: `a1bee4c` by Wisit Suwannao
- Prev: `2d288c71e13b` | Δrows vs prev: 20,530 (79.3%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 10) (#38)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-13 – `2d288c71e13b` – #37

- Commit: `bc06d1f` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,530 (79.3%)
- ↩ Same file as 2026-11-12 `8fd32a7` (#35)
- Why: feat: Run 10 SOTA elevation with 2x2 Quadrant GBDT (Task 2: 0.7524) and 5-Paradigm Deep Autoencoder (Task 5: F3 0.90+) (#37)
- Merged: `3e418d058152` | Kaggle public: 0.72682

### 2026-11-12 – `541099d9e792` – #36

- Commit: `485cf48` by Wisit Suwannao
- Prev: `2d288c71e13b` | Δrows vs prev: 20,530 (79.3%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 9) (#36)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `2d288c71e13b` – #35

- Commit: `8fd32a7` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 20,530 (79.3%)
- Why: feat: Run 9 SOTA elevation with new TA dataset (Task 2 Cohort GBDT 0.7508, Task 4 ConvNeXt-Tiny MLP 0.9019, Task 5 F3 Anomaly) (#35)
- Merged: `f71ce3e2349b` | Kaggle public: 0.71770

### 2026-11-12 – `541099d9e792` – #34

- Commit: `af4c26f` by Wisit Suwannao
- Prev: `e55c20b3dea3` | Δrows vs prev: 7,234 (27.9%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 8) (#34)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `e55c20b3dea3` – #33

- Commit: `470ce3c` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 7,234 (27.9%)
- ↩ Same file as 2026-11-10 `ecca1dc` (#22)
- Why: feat: elevate Tasks 1-5 with Bayes-Optimal Dynamic Quantile Hurdle and SOTA Ensembles (Run 8) (#33)
- Merged: `2355d36d5abe` | Kaggle public: 0.42367

### 2026-11-12 – `541099d9e792` – #32

- Commit: `a23848d` by Wisit Suwannao
- Prev: `e55c20b3dea3` | Δrows vs prev: 7,234 (27.9%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#32) (#32)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `e55c20b3dea3` – #31

- Commit: `a78540d` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 7,234 (27.9%)
- ↩ Same file as 2026-11-10 `ecca1dc` (#22)
- Why: feat: SOTA Run 7 elevation (Task 5 PyOD PCA 0.05 contamination + Task 2 Cohort 0.966 + Task 4 DINOv2) (#31)
- Merged: `bc8836f51f7d` | Kaggle public: 0.67549

### 2026-11-12 – `541099d9e792` – #30

- Commit: `f00d286` by Wisit Suwannao
- Prev: `e55c20b3dea3` | Δrows vs prev: 7,234 (27.9%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#30) (#30)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `e55c20b3dea3` – #29

- Commit: `2b8b419` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 7,234 (27.9%)
- ↩ Same file as 2026-11-10 `ecca1dc` (#22)
- Why: feat: SOTA Run 6 champion elevation (Task 4 DINOv2 0.947, Task 2 cohort 0.966, Task 5 F3 tripwires) (#29)
- Merged: `ba9f3a6f82b1` | Kaggle public: 0.71761

### 2026-11-12 – `541099d9e792` – #28

- Commit: `27a8a04` by Wisit Suwannao
- Prev: `24430d0a1d88` | Δrows vs prev: 10,727 (41.4%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#28)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `24430d0a1d88` – #27

- Commit: `456a7d5` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 10,727 (41.4%)
- Why: feat: SOTA Run 5 elevation (Task 2 cohort GBDT 0.966, Task 4 DINOv2 0.916, Task 5 F3 ensemble) (#27)
- Merged: `aaa52adb72f8` | Kaggle public: 0.65531

### 2026-11-10 – `541099d9e792` – #26

- Commit: `f19961b` by Wisit Suwannao
- Prev: `0b3055aa71ce` | Δrows vs prev: 8,268 (31.9%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#26)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-10 – `0b3055aa71ce` – #25

- Commit: `24ce629` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 8,268 (31.9%)
- ↩ Same file as 2026-11-07 `bdb7de6` (#17)
- Why: feat: SOTA run 4 elevation targeting >=0.80 benchmark score (#25)
- Merged: `5eca03fe5225` | Kaggle public: 0.67352

### 2026-11-10 – `541099d9e792` – #23

- Commit: `8d183dc` by Wisit Suwannao
- Prev: `e55c20b3dea3` | Δrows vs prev: 7,234 (27.9%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#23)
- Merged: `a1623e0ddc82` | Kaggle public: –

### 2026-11-10 – `e55c20b3dea3` – #22

- Commit: `ecca1dc` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 7,234 (27.9%)
- Why: feat: test submission (Task 3 Hurdle + Task 4 numeric 0-4) (#22)
- Merged: `e0c9e62e2d39` | Kaggle public: 0.51909

### 2026-11-07 – `541099d9e792` – #18

- Commit: `82b824f` by Wisit Suwannao
- Prev: `0b3055aa71ce` | Δrows vs prev: 8,268 (31.9%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#18)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `0b3055aa71ce` – #17

- Commit: `bdb7de6` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 8,268 (31.9%)
- Why: feat: SOTA run 3 with calibrated DINOv2 and high-confidence anomaly tripwires (#17)
- Merged: `e80019f8da94` | Kaggle public: 0.63573

### 2026-11-07 – `541099d9e792` – #16

- Commit: `2d82be0` by Wisit Suwannao
- Prev: `2e9a42d2c31c` | Δrows vs prev: 8,140 (31.4%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#16)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `2e9a42d2c31c` – #15

- Commit: `e89f219` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 8,140 (31.4%)
- Why: feat: elevate all 5 tasks targeting >=0.80 score with DINOv2 and F3 calibration (#15)
- Merged: `d36e163f2f81` | Kaggle public: 0.63286

### 2026-11-07 – `541099d9e792` – #14

- Commit: `d96af9d` by Wisit Suwannao
- Prev: `08ca6bee81c4` | Δrows vs prev: 12,326 (47.6%)
- ↩ Same file as 2026-11-05 `b6c5d61` (#10)
- Why: chore: restore teammate tasks to upstream state post-submission (#14)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-06 – `08ca6bee81c4` – #13

- Commit: `0650975` by Wisit Suwannao
- Prev: `541099d9e792` | Δrows vs prev: 12,326 (47.6%)
- Why: feat: elevate all 5 tasks (Task 1-5) beating baseline with SOTA models (#13)
- Merged: `8adba6602333` | Kaggle public: 0.63573

### 2026-11-05 – `541099d9e792` – #10

- Commit: `b6c5d61` by madmax16_
- Prev: `906c088efe5e` | Δrows vs prev: 21,583 (83.4%)
- Why: feat(task4): add game title image classification pipeline (#10)
- Merged: `881d6d9c1cff` | Kaggle public: 0.47978

### 2026-11-03 – `906c088efe5e` – #9

- Commit: `f552106` by Wisit Suwannao
- Prev: `7d4297fe5ce1` | Δrows vs prev: 15,792 (61.0%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore(tasks-2-4-5): reset secondary tasks to scaffolding for team contributors (#9)
- Merged: `b6a38ac93a15` | Kaggle public: –

### 2026-11-03 – `7d4297fe5ce1` – #7

- Commit: `ba483f9` by Wisit Suwannao
- Prev: `906c088efe5e` | Δrows vs prev: 15,792 (61.0%)
- Why: feat(tasks-2-4-5): elevate secondary tasks with trained ensemble models and test predictions (#7)
- Merged: `4891205ca6cc` | Kaggle public: 0.58091

### 2026-11-02 – `906c088efe5e` – #2

- Commit: `e6226cf` by bbb
- Prev: – | Δrows vs prev: new file
- Why: ci: add submission merge check and kaggle submit workflows (#2)
- Merged: `48a77166cd56` | Kaggle public: 0.12007
