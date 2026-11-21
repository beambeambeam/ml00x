# Task 2 submission (`submit/index.csv`)

> Agents keep this file. Rules: `AGENTS.md` → Submission history. Newest entry first. Never edit or delete old changelog entries.

## Current

- **sha12:** `34d927a40901` (2026-11-21, source `2f988af`, #79)
- **Active Version:** **Run 22 models with corrected affine macro-F1 export**
- **Generating Script:** `task2/run_bayes_export.py`; run `uv run python run_bayes_export.py --promote` from `task2/`.
- **Generating Notebook:** `task2/task2_player_segment.ipynb`; its single executable export calls the same `predict_segments` function as validation.
- **Required Artifacts:**
  - `task2/predictions/model-level/task2-exact-param-em-bundle.npz`
  - `task2/predictions/model-level/segment-tuned-catboost-lightgbm-revised-2026-11-12-macro-f1-0.751314-oof.csv`
  - `task2/predictions/model-level/segment-tuned-catboost-lightgbm-revised-2026-11-12-macro-f1-0.751314-test-predictions.csv`
  - `task2/data/train.csv`, `task2/data/test.csv`
- **Outputs:** `task2/predictions/bayes-export/{oof,test,index}.csv`, `task2/models/bayes-export/decision-rule.json`, `task2/assets/bayes-export/`.
- **OOF Metrics:**
  - Corrected affine export macro F1 **`0.759316`**, versus **`0.753158`** for the previous executable export rule. Weighted F1 `0.771085` versus `0.763171`.
  - All five folds improve; paired row-bootstrap 95% interval for macro-F1 gain: `[+0.004653, +0.007692]`.
  - `OOF LogLoss = 0.595646`; existing log-loss stack and fixed-point decision fit are retained.
- **Validation limits:** Cross-fitted postprocessing of frozen OOF probabilities, not fully nested base-model retraining. Other OOF models can have seen held-out postprocessing-fold labels. Bootstrap is conditional on these models and folds. The NPZ lacks row IDs; original order is assumed, input hashes are recorded, and keyed-model labels are checked. Corrected export has no Kaggle score yet.
- **Test Distribution (`25,889` rows):** `0`: `10,558` (`40.8%`), `1`: `6,583` (`25.4%`), `2`: `5,046` (`19.5%`), `3`: `3,702` (`14.3%`)

## Changelog

### 2026-11-21 – `34d927a40901` – #79

- Commit: `2f988af` by bbb
- Prev: `d8273939dc94` | Δrows vs prev: 1,622 (6.3%)
- Why: `run_bayes_export.py` reproduces the previous submission with the multiplier-only rule, then restores the validated `alpha * p - beta` rule. The correct export was a Markdown cell; the executable cell omitted beta. On the same frozen OOF models, macro F1 increases from `0.753158` to `0.759316`, with gains on all five postprocessing folds. The score is a replay comparison, not a new fully nested training result.
- Merged: `904a1fd0976c` | Kaggle public: –

### 2026-11-20 – `d8273939dc94` – #69

- Commit: `8654961` by bbb
- Prev: `f180b8058929` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: fix: restore Run 22 task2 and task4 submit/index.csv (#69)
- Merged: `296e7a3ae27a` | Kaggle public: 0.84113

### 2026-11-20 – `f180b8058929` – #66

- Commit: `52fc1e3` by Wisit Suwannao
- Prev: `d8273939dc94` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 22 Kaggle submission (#66)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `d8273939dc94` – #65

- Commit: `5014990` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 22 Task 4 Pure-Train Uniform-Prior Debias(a=0.860) + DINOv2-StageC (#65)
- Merged: `b45d08b2eabc` | Kaggle public: 0.86337

### 2026-11-20 – `f180b8058929` – #64

- Commit: `b68ac80` by Wisit Suwannao
- Prev: `d8273939dc94` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 21 Kaggle submission (#64)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `d8273939dc94` – #63

- Commit: `5ff5a3d` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 21 Task 4 5-Fold SGKF DINOv2-StageC Nightmare-Hardened (#63)
- Merged: `95d9db7f8078` | Kaggle public: 0.86146

### 2026-11-19 – `f180b8058929` – #62

- Commit: `987486f` by Wisit Suwannao
- Prev: `d8273939dc94` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 20 Kaggle submission (#62)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `d8273939dc94` – #61

- Commit: `ca1bd6a` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 20 Task 4 pure-train Hard+Iso Nested SGKF + Train-Prior debias (#61)
- Merged: `820b12571d07` | Kaggle public: 0.86153

### 2026-11-19 – `f180b8058929` – #60

- Commit: `8f040db` by Wisit Suwannao
- Prev: `d8273939dc94` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 19-C Kaggle submission (#60)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `d8273939dc94` – #59

- Commit: `796efb1` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 19-C strict zero-test-parameter compliance across Tasks 1 & 4 (#59)
- Merged: `42b91194dc33` | Kaggle public: 0.85811

### 2026-11-19 – `f180b8058929` – #58

- Commit: `c231933` by Wisit Suwannao
- Prev: `d8273939dc94` | Δrows vs prev: 18,473 (71.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 18 Kaggle submission (#58)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `d8273939dc94` – #57

- Commit: `5acf8b8` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,473 (71.4%)
- Why: feat: Run 18 100% Pure-Train DGP Elevation across Tasks 1-5 (#57)
- Merged: `3f236f5e932e` | Kaggle public: 0.85908

### 2026-11-17 – `f180b8058929` – #55

- Commit: `38b3bab` by Wisit Suwannao
- Prev: `43677422a9ed` | Δrows vs prev: 18,224 (70.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 17 Kaggle submission (#55)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-17 – `43677422a9ed` – #54

- Commit: `50874df` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,224 (70.4%)
- ↩ Same file as 2026-11-16 `c4b21da` (#50)
- Why: feat(run17): upgrade Task 4 5-Fold SGKF E2E+ViT Stack & Task 5 Gaussian/NaN fix (#54)
- Merged: `005367eb3fed` | Kaggle public: 0.84139

### 2026-11-16 – `f180b8058929` – #51

- Commit: `f64ac9a` by Wisit Suwannao
- Prev: `43677422a9ed` | Δrows vs prev: 18,224 (70.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 16 Kaggle submission (#51)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-16 – `43677422a9ed` – #50

- Commit: `c4b21da` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,224 (70.4%)
- Why: feat(run16): upgrade Task 3 V3+MPS & Task 5 4-Fam Raw-NaN + Run 15 T1/T2/T4 (#50)
- Merged: `d57456bd72f3` | Kaggle public: 0.83969

### 2026-11-15 – `f180b8058929` – #47

- Commit: `c0b808a` by Wisit Suwannao
- Prev: `cbd73344792c` | Δrows vs prev: 18,249 (70.5%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 14 Kaggle submission (#47)
- Merged: `7a843b787104` | Kaggle public: 0.59607

### 2026-11-15 – `cbd73344792c` – #46

- Commit: `c7dc931` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,249 (70.5%)
- Why: feat(run14): upgrade Tasks 2 & 3 zero-leakage nested ensembles + Task 4 4xViT+SWA stack (#46)
- Merged: `1362e3d02daa` | Kaggle public: 0.83904

### 2026-11-14 – `f180b8058929` – #45

- Commit: `4cb35e5` by Wisit Suwannao
- Prev: `975c12498a71` | Δrows vs prev: 18,263 (70.5%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Kaggle submission (#45)
- Merged: `ea7b3897e3cd` | Kaggle public: –

### 2026-11-14 – `975c12498a71` – #44

- Commit: `997bcd2` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,263 (70.5%)
- Why: feat(run12): upgrade all 5 tasks + Task 3 dual-grid Bayes quantile hurdle (#44)
- Merged: `1727f0c63c80` | Kaggle public: 0.81773

### 2026-11-14 – `f180b8058929` – #43

- Commit: `47239c2` by Wisit Suwannao
- Prev: `5943134e506c` | Δrows vs prev: 18,230 (70.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11 retry) (#43)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-14 – `5943134e506c` – #42

- Commit: `d4f7793` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,230 (70.4%)
- ↩ Same file as 2026-11-13 `9bc7502` (#40)
- Why: feat: Run 11 SOTA leak-free elevation (retry post daily reset) (#42)
- Merged: `8b5c4a86390e` | Kaggle public: 0.81491

### 2026-11-13 – `f180b8058929` – #41

- Commit: `411aae1` by Wisit Suwannao
- Prev: `5943134e506c` | Δrows vs prev: 18,230 (70.4%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11) (#41)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-13 – `5943134e506c` – #40

- Commit: `9bc7502` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,230 (70.4%)
- Why: feat: Run 11 SOTA leak-free elevation (Task 3 BayesQ 0.7952, Task 4 SWA 0.9567, Tasks 1-2 & 5) (#40)
- Merged: `8b5c4a86390e` | Kaggle public: –

### 2026-11-13 – `f180b8058929` – #38

- Commit: `a1bee4c` by Wisit Suwannao
- Prev: `1a0ac6452330` | Δrows vs prev: 18,177 (70.2%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 10) (#38)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-13 – `1a0ac6452330` – #37

- Commit: `bc06d1f` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,177 (70.2%)
- Why: feat: Run 10 SOTA elevation with 2x2 Quadrant GBDT (Task 2: 0.7524) and 5-Paradigm Deep Autoencoder (Task 5: F3 0.90+) (#37)
- Merged: `3e418d058152` | Kaggle public: 0.72682

### 2026-11-12 – `f180b8058929` – #36

- Commit: `485cf48` by Wisit Suwannao
- Prev: `175a89e5034d` | Δrows vs prev: 18,132 (70.0%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 9) (#36)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `175a89e5034d` – #35

- Commit: `8fd32a7` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 18,132 (70.0%)
- Why: feat: Run 9 SOTA elevation with new TA dataset (Task 2 Cohort GBDT 0.7508, Task 4 ConvNeXt-Tiny MLP 0.9019, Task 5 F3 Anomaly) (#35)
- Merged: `f71ce3e2349b` | Kaggle public: 0.71770

### 2026-11-12 – `f180b8058929` – #32

- Commit: `a23848d` by Wisit Suwannao
- Prev: `0d319487a8a3` | Δrows vs prev: 5,459 (21.1%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (#32) (#32)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `0d319487a8a3` – #31

- Commit: `a78540d` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 5,459 (21.1%)
- ↩ Same file as 2026-11-12 `456a7d5` (#27)
- Why: feat: SOTA Run 7 elevation (Task 5 PyOD PCA 0.05 contamination + Task 2 Cohort 0.966 + Task 4 DINOv2) (#31)
- Merged: `bc8836f51f7d` | Kaggle public: 0.67549

### 2026-11-12 – `f180b8058929` – #30

- Commit: `f00d286` by Wisit Suwannao
- Prev: `0d319487a8a3` | Δrows vs prev: 5,459 (21.1%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (#30) (#30)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `0d319487a8a3` – #29

- Commit: `2b8b419` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 5,459 (21.1%)
- ↩ Same file as 2026-11-12 `456a7d5` (#27)
- Why: feat: SOTA Run 6 champion elevation (Task 4 DINOv2 0.947, Task 2 cohort 0.966, Task 5 F3 tripwires) (#29)
- Merged: `ba9f3a6f82b1` | Kaggle public: 0.71761

### 2026-11-12 – `f180b8058929` – #28

- Commit: `27a8a04` by Wisit Suwannao
- Prev: `0d319487a8a3` | Δrows vs prev: 5,459 (21.1%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (#28)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `0d319487a8a3` – #27

- Commit: `456a7d5` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 5,459 (21.1%)
- Why: feat: SOTA Run 5 elevation (Task 2 cohort GBDT 0.966, Task 4 DINOv2 0.916, Task 5 F3 ensemble) (#27)
- Merged: `aaa52adb72f8` | Kaggle public: 0.65531

### 2026-11-10 – `f180b8058929` – #26

- Commit: `f19961b` by Wisit Suwannao
- Prev: `a5bb6f51b640` | Δrows vs prev: 1,294 (5.0%)
- ↩ Same file as 2026-11-10 `ff6a94f` (#21)
- Why: chore: restore teammate tasks to upstream state post-submission (#26)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-10 – `a5bb6f51b640` – #25

- Commit: `24ce629` by Wisit Suwannao
- Prev: `f180b8058929` | Δrows vs prev: 1,294 (5.0%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA run 4 elevation targeting >=0.80 benchmark score (#25)
- Merged: `5eca03fe5225` | Kaggle public: 0.67352

### 2026-11-10 – `f180b8058929` – #21

- Commit: `ff6a94f` by Guntee Doungmanee
- Prev: `a5bb6f51b640` | Δrows vs prev: 1,294 (5.0%)
- Why: fix(task2): submit RealMLP-TD predictions as new best index.csv (#21)
- Merged: `e7ee94c8314e` | Kaggle public: –

### 2026-11-09 – `a5bb6f51b640` – #19

- Commit: `f34c92b` by Guntee Doungmanee
- Prev: `771b6e977501` | Δrows vs prev: 448 (1.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: fix(task2): restore top-pick submission predictions (public 0.58091) (#19)
- Merged: `5dc3d5a113e8` | Kaggle public: 0.48054

### 2026-11-07 – `771b6e977501` – #18

- Commit: `82b824f` by Wisit Suwannao
- Prev: `a5bb6f51b640` | Δrows vs prev: 448 (1.7%)
- ↩ Same file as 2026-11-03 `a286632` (#11)
- Why: chore: restore teammate tasks to upstream state post-submission (#18)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `a5bb6f51b640` – #17

- Commit: `bdb7de6` by Wisit Suwannao
- Prev: `771b6e977501` | Δrows vs prev: 448 (1.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA run 3 with calibrated DINOv2 and high-confidence anomaly tripwires (#17)
- Merged: `e80019f8da94` | Kaggle public: 0.63573

### 2026-11-07 – `771b6e977501` – #16

- Commit: `2d82be0` by Wisit Suwannao
- Prev: `a5bb6f51b640` | Δrows vs prev: 448 (1.7%)
- ↩ Same file as 2026-11-03 `a286632` (#11)
- Why: chore: restore teammate tasks to upstream state post-submission (#16)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `a5bb6f51b640` – #15

- Commit: `e89f219` by Wisit Suwannao
- Prev: `771b6e977501` | Δrows vs prev: 448 (1.7%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: elevate all 5 tasks targeting >=0.80 score with DINOv2 and F3 calibration (#15)
- Merged: `d36e163f2f81` | Kaggle public: 0.63286

### 2026-11-07 – `771b6e977501` – #14

- Commit: `d96af9d` by Wisit Suwannao
- Prev: `a5bb6f51b640` | Δrows vs prev: 448 (1.7%)
- ↩ Same file as 2026-11-03 `a286632` (#11)
- Why: chore: restore teammate tasks to upstream state post-submission (#14)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-06 – `a5bb6f51b640` – #13

- Commit: `0650975` by Wisit Suwannao
- Prev: `771b6e977501` | Δrows vs prev: 448 (1.7%)
- Why: feat: elevate all 5 tasks (Task 1-5) beating baseline with SOTA models (#13)
- Merged: `8adba6602333` | Kaggle public: 0.63573

### 2026-11-03 – `771b6e977501` – #11

- Commit: `a286632` by Guntee Doungmanee
- Prev: `54fb060365ca` | Δrows vs prev: 14,611 (56.4%)
- Why: feat(task2): add segment research artifacts and top model (#11)
- Merged: `b8853c7b0557` | Kaggle public: 0.47978

### 2026-11-03 – `54fb060365ca` – #9

- Commit: `f552106` by Wisit Suwannao
- Prev: `5710f4247950` | Δrows vs prev: 15,373 (59.4%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore(tasks-2-4-5): reset secondary tasks to scaffolding for team contributors (#9)
- Merged: `b6a38ac93a15` | Kaggle public: –

### 2026-11-03 – `5710f4247950` – #7

- Commit: `ba483f9` by Wisit Suwannao
- Prev: `54fb060365ca` | Δrows vs prev: 15,373 (59.4%)
- Why: feat(tasks-2-4-5): elevate secondary tasks with trained ensemble models and test predictions (#7)
- Merged: `4891205ca6cc` | Kaggle public: 0.58091

### 2026-11-02 – `54fb060365ca` – #2

- Commit: `e6226cf` by bbb
- Prev: – | Δrows vs prev: new file
- Why: ci: add submission merge check and kaggle submit workflows (#2)
- Merged: `48a77166cd56` | Kaggle public: 0.12007
