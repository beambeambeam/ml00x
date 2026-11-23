# Task 5 submission (`submit/index.csv`)

> Agents keep this file. Rules: `AGENTS.md` → Submission history. Newest entry first. Never edit or delete old changelog entries.

## Current

- **sha12:** `a5005a73a89d` (since 2026-11-23, #86)
- **Active Version:** **Archetype rules + zero-parameter Exact Analytical Longitudinal Bayes (`ExactBayesUB`)** (`Palapluem`, PR `#86`)
- **Generating Script:** `task5/archetype_rules.py` (`uv run archetype_rules.py`)
- **Required Input:** `task5/data/test.csv` (`25,889` accounts × `123` columns)
- **Outputs:** `task5/predictions/archetype-rules.csv`, `task5/assets/archetype-rule-counts.csv`, `task5/assets/erratic-combat-box.png`, `task5/assets/unstable-nan-calibration.csv`, `task5/assets/unstable-nan-cut-expected-f3.csv`
- **Detector Components:**
  - Exact uniform-box rules for 8 injected archetypes (account_takeover, vpn_pricing_abuse, account_sale_boosting, bot_24_7, low_activity, silent_account, unstable_behavior, erratic_combat): `2,052` flags
  - NaN `behavior_stability` recovery (`ExactBayesUB`): hard filter (all `purchase_frequency` in [3, 6], age ≥ 795) + closed-form Normal marginal density over `purchase_frequency_1..4` ($B_{\text{pf}} \sim U(0,5)$), `login_count_1..4` ($B_{\text{lc}} \sim U(20,100)$), and `account_age_days` ($U(180,1500)$) vs. UB i.i.d. `purchase_frequency` ($U(3,6)$), product-Uniform `login_count` ($m_p \sim U(0.40, 0.70)$, 200-pt Gauss-Legendre), and `account_age_days` ($U(800,1500)$), cut `0.11513` (`0.115`) by expected F3: `+25` flags (`Expected F3 = 0.99835`)
  - Masked OOF check on `22,926` players (`known & ~sure_7`, `2,393` hard-filter candidates, `246` positives): `AUC 0.91251 → 0.93656` (`22,926` full `0.99169 → 0.99399`), `PR-AUC 0.69322 → 0.79922`, `LogLoss 0.18256 → 0.13851` (`22,926` full `0.01866 → 0.01446`), `10·TP − Flags 1,394 → 1,502` (`Boot P = 0.9892`), passes two-way stratified split-half replication (`Half A AUC=0.92757, PR=0.77952`; `Half B AUC=0.94609, PR=0.81982`)
- **Test Distribution (`25,889` rows):** `0` (normal): `23,812` (`92.0%`), `1` (anomalous): `2,077` (`8.0%`)

## Changelog

### 2026-11-23 – `a5005a73a89d` – #86

- Commit: `ca30f19` by Palapluem (Wisit Suwannao)
- Prev: `b308313ff651` | Δrows vs prev: 13 (0.1%)
- Why: feat(task5): replace LightGBM NaN-unstable recovery with zero-parameter Exact Analytical Longitudinal Bayes integral (`ExactBayesUB`) in `task5/archetype_rules.py`; `22,926` masked OOF (`2,393` hard candidates, `246` pos) `AUC 0.91251 → 0.93656`, `PR-AUC 0.69322 → 0.79922`, `LogLoss 0.18256 → 0.13851`, `10·TP−Flags 1,394 → 1,502` (`Boot P=0.9892`), `Expected F3 = 0.99835`, `+25` NaN flags (`2,077` total, `+4/-9` vs #82, `+6/-5` vs #71)
- Merged: `f5ad4958942c` | Kaggle public: – (combined in `e3b50d7d48a7`: 0.86545)

### 2026-11-22 – `b308313ff651` – #82

- Commit: `571aefb` by Palapluem (Wisit Suwannao)
- Prev: `5e57282c6459` | Δrows vs prev: 10 (0.0%)
- Why: feat(task5): add stability features (`lc_std`, `lc_range`, `pf_range`, `age_gt_800`, `lc_norm_age`) and 50/50 Platt+Isotonic calibration to `unstable_nan_probability` (`task5/archetype_rules.py`); nested `22,926` masked OOF `AUC 0.89795→0.91046`, `PR 0.62878→0.68293`, `LL 0.20529→0.18360`, `10·TP−flags 1343→1395` (`Boot P=0.911`), wins both `A→B` and `B→A` split-half replications; `Expected F3 = 0.99790`, `+30` NaN flags (`+8/-2` vs #71)
- Merged: `e952e560ec4c` | Kaggle public: 0.86530

### 2026-11-20 – `5e57282c6459` – #71

- Commit: `05a762a` by bbb
- Prev: `92253faba1f6` | Δrows vs prev: 19 (0.1%)
- Why: feat: Task 5 archetype rules with calibrated NaN-unstable recovery (#71). `task5/archetype_rules.py`; expected F3 0.99785 vs 0.99744 for Run 22; removes 16 low-probability NaN-unstable flags, adds 3
- Merged: `1f26381be9b0` | Kaggle public: –

### 2026-11-20 – `92253faba1f6` – #70

- Commit: `33160ae` by bbb
- Prev: `6799912e28a4` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: fix: restore Run 22 task5 submit/index.csv (#70)
- Merged: `b3ea360f52c3` | Kaggle public: 0.86363

### 2026-11-20 – `6799912e28a4` – #66

- Commit: `52fc1e3` by Wisit Suwannao
- Prev: `92253faba1f6` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 22 Kaggle submission (#66)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `92253faba1f6` – #65

- Commit: `5014990` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 22 Task 4 Pure-Train Uniform-Prior Debias(a=0.860) + DINOv2-StageC (#65)
- Merged: `b45d08b2eabc` | Kaggle public: 0.86337

### 2026-11-20 – `6799912e28a4` – #64

- Commit: `b68ac80` by Wisit Suwannao
- Prev: `92253faba1f6` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 21 Kaggle submission (#64)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-20 – `92253faba1f6` – #63

- Commit: `5ff5a3d` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 21 Task 4 5-Fold SGKF DINOv2-StageC Nightmare-Hardened (#63)
- Merged: `95d9db7f8078` | Kaggle public: 0.86146

### 2026-11-19 – `6799912e28a4` – #62

- Commit: `987486f` by Wisit Suwannao
- Prev: `92253faba1f6` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 20 Kaggle submission (#62)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `92253faba1f6` – #61

- Commit: `ca1bd6a` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 20 Task 4 pure-train Hard+Iso Nested SGKF + Train-Prior debias (#61)
- Merged: `820b12571d07` | Kaggle public: 0.86153

### 2026-11-19 – `6799912e28a4` – #60

- Commit: `8f040db` by Wisit Suwannao
- Prev: `92253faba1f6` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 19-C Kaggle submission (#60)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `92253faba1f6` – #59

- Commit: `796efb1` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-19 `5acf8b8` (#57)
- Why: feat: Run 19-C strict zero-test-parameter compliance across Tasks 1 & 4 (#59)
- Merged: `42b91194dc33` | Kaggle public: 0.85811

### 2026-11-19 – `6799912e28a4` – #58

- Commit: `c231933` by Wisit Suwannao
- Prev: `92253faba1f6` | Δrows vs prev: 291 (1.1%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate task1,2,4,5 submit/index.csv after Run 18 Kaggle submission (#58)
- Merged: `f78cc560de14` | Kaggle public: –

### 2026-11-19 – `92253faba1f6` – #57

- Commit: `5acf8b8` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 291 (1.1%)
- Why: feat: Run 18 100% Pure-Train DGP Elevation across Tasks 1-5 (#57)
- Merged: `3f236f5e932e` | Kaggle public: 0.85908

### 2026-11-17 – `6799912e28a4` – #55

- Commit: `38b3bab` by Wisit Suwannao
- Prev: `2a5c502d88a9` | Δrows vs prev: 47 (0.2%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 17 Kaggle submission (#55)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-17 – `2a5c502d88a9` – #54

- Commit: `50874df` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 47 (0.2%)
- Why: feat(run17): upgrade Task 4 5-Fold SGKF E2E+ViT Stack & Task 5 Gaussian/NaN fix (#54)
- Merged: `005367eb3fed` | Kaggle public: 0.84139

### 2026-11-16 – `6799912e28a4` – #51

- Commit: `f64ac9a` by Wisit Suwannao
- Prev: `bf93741562fd` | Δrows vs prev: 86 (0.3%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 16 Kaggle submission (#51)
- Merged: `ac42bd0a2e56` | Kaggle public: –

### 2026-11-16 – `bf93741562fd` – #50

- Commit: `c4b21da` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 86 (0.3%)
- Why: feat(run16): upgrade Task 3 V3+MPS & Task 5 4-Fam Raw-NaN + Run 15 T1/T2/T4 (#50)
- Merged: `d57456bd72f3` | Kaggle public: 0.83969

### 2026-11-15 – `6799912e28a4` – #47

- Commit: `c0b808a` by Wisit Suwannao
- Prev: `26f445797fe7` | Δrows vs prev: 71 (0.3%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Run 14 Kaggle submission (#47)
- Merged: `7a843b787104` | Kaggle public: 0.59607

### 2026-11-15 – `26f445797fe7` – #46

- Commit: `c7dc931` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 71 (0.3%)
- ↩ Same file as 2026-11-14 `997bcd2` (#44)
- Why: feat(run14): upgrade Tasks 2 & 3 zero-leakage nested ensembles + Task 4 4xViT+SWA stack (#46)
- Merged: `1362e3d02daa` | Kaggle public: 0.83904

### 2026-11-14 – `6799912e28a4` – #45

- Commit: `4cb35e5` by Wisit Suwannao
- Prev: `26f445797fe7` | Δrows vs prev: 71 (0.3%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate tasks (task1, task2, task4, task5) after Kaggle submission (#45)
- Merged: `ea7b3897e3cd` | Kaggle public: –

### 2026-11-14 – `26f445797fe7` – #44

- Commit: `997bcd2` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 71 (0.3%)
- Why: feat(run12): upgrade all 5 tasks + Task 3 dual-grid Bayes quantile hurdle (#44)
- Merged: `1727f0c63c80` | Kaggle public: 0.81773

### 2026-11-14 – `6799912e28a4` – #43

- Commit: `47239c2` by Wisit Suwannao
- Prev: `438846585ff5` | Δrows vs prev: 20 (0.1%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11 retry) (#43)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-14 – `438846585ff5` – #42

- Commit: `d4f7793` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 20 (0.1%)
- ↩ Same file as 2026-11-13 `9bc7502` (#40)
- Why: feat: Run 11 SOTA leak-free elevation (retry post daily reset) (#42)
- Merged: `8b5c4a86390e` | Kaggle public: 0.81491

### 2026-11-13 – `6799912e28a4` – #41

- Commit: `411aae1` by Wisit Suwannao
- Prev: `438846585ff5` | Δrows vs prev: 20 (0.1%)
- ↩ Same file as 2026-11-13 `2602a80` (#39)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 11) (#41)
- Merged: `c88d74f878e6` | Kaggle public: –

### 2026-11-13 – `438846585ff5` – #40

- Commit: `9bc7502` by Wisit Suwannao
- Prev: `6799912e28a4` | Δrows vs prev: 20 (0.1%)
- Why: feat: Run 11 SOTA leak-free elevation (Task 3 BayesQ 0.7952, Task 4 SWA 0.9567, Tasks 1-2 & 5) (#40)
- Merged: `8b5c4a86390e` | Kaggle public: –

### 2026-11-13 – `6799912e28a4` – #39

- Commit: `2602a80` by Natthawat Primsirikunawut
- Prev: `aad484def371` | Δrows vs prev: 1,818 (7.0%)
- Why: feat:  task5 add account security anomaly detection pipeline (#39)
- Merged: `35eeed2758fd` | Kaggle public: 0.59580

### 2026-11-13 – `aad484def371` – #38

- Commit: `a1bee4c` by Wisit Suwannao
- Prev: `57304c490053` | Δrows vs prev: 3,754 (14.5%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 10) (#38)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-13 – `57304c490053` – #37

- Commit: `bc06d1f` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 3,754 (14.5%)
- Why: feat: Run 10 SOTA elevation with 2x2 Quadrant GBDT (Task 2: 0.7524) and 5-Paradigm Deep Autoencoder (Task 5: F3 0.90+) (#37)
- Merged: `3e418d058152` | Kaggle public: 0.72682

### 2026-11-12 – `aad484def371` – #36

- Commit: `485cf48` by Wisit Suwannao
- Prev: `3e05e499835a` | Δrows vs prev: 2,149 (8.3%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 9) (#36)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `3e05e499835a` – #35

- Commit: `8fd32a7` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,149 (8.3%)
- Why: feat: Run 9 SOTA elevation with new TA dataset (Task 2 Cohort GBDT 0.7508, Task 4 ConvNeXt-Tiny MLP 0.9019, Task 5 F3 Anomaly) (#35)
- Merged: `f71ce3e2349b` | Kaggle public: 0.71770

### 2026-11-12 – `aad484def371` – #34

- Commit: `af4c26f` by Wisit Suwannao
- Prev: `35573ea292e0` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (Run 8) (#34)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `35573ea292e0` – #33

- Commit: `470ce3c` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: elevate Tasks 1-5 with Bayes-Optimal Dynamic Quantile Hurdle and SOTA Ensembles (Run 8) (#33)
- Merged: `2355d36d5abe` | Kaggle public: 0.42367

### 2026-11-12 – `aad484def371` – #32

- Commit: `a23848d` by Wisit Suwannao
- Prev: `bc72e2dd1dcc` | Δrows vs prev: 1,295 (5.0%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (#32) (#32)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `bc72e2dd1dcc` – #31

- Commit: `a78540d` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 1,295 (5.0%)
- Why: feat: SOTA Run 7 elevation (Task 5 PyOD PCA 0.05 contamination + Task 2 Cohort 0.966 + Task 4 DINOv2) (#31)
- Merged: `bc8836f51f7d` | Kaggle public: 0.67549

### 2026-11-12 – `aad484def371` – #30

- Commit: `f00d286` by Wisit Suwannao
- Prev: `35573ea292e0` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (#30) (#30)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `35573ea292e0` – #29

- Commit: `2b8b419` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA Run 6 champion elevation (Task 4 DINOv2 0.947, Task 2 cohort 0.966, Task 5 F3 tripwires) (#29)
- Merged: `ba9f3a6f82b1` | Kaggle public: 0.71761

### 2026-11-12 – `aad484def371` – #28

- Commit: `27a8a04` by Wisit Suwannao
- Prev: `7f6a1deebaca` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (#28)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-12 – `7f6a1deebaca` – #27

- Commit: `456a7d5` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,150 (8.3%)
- Why: feat: SOTA Run 5 elevation (Task 2 cohort GBDT 0.966, Task 4 DINOv2 0.916, Task 5 F3 ensemble) (#27)
- Merged: `aaa52adb72f8` | Kaggle public: 0.65531

### 2026-11-10 – `aad484def371` – #26

- Commit: `f19961b` by Wisit Suwannao
- Prev: `35573ea292e0` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (#26)
- Merged: `06683c0cfd15` | Kaggle public: –

### 2026-11-10 – `35573ea292e0` – #25

- Commit: `24ce629` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA run 4 elevation targeting >=0.80 benchmark score (#25)
- Merged: `5eca03fe5225` | Kaggle public: 0.67352

### 2026-11-07 – `aad484def371` – #18

- Commit: `82b824f` by Wisit Suwannao
- Prev: `35573ea292e0` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (#18)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `35573ea292e0` – #17

- Commit: `bdb7de6` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-06 `0650975` (#13)
- Why: feat: SOTA run 3 with calibrated DINOv2 and high-confidence anomaly tripwires (#17)
- Merged: `e80019f8da94` | Kaggle public: 0.63573

### 2026-11-07 – `aad484def371` – #16

- Commit: `2d82be0` by Wisit Suwannao
- Prev: `38615b932cbe` | Δrows vs prev: 3,500 (13.5%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (#16)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-07 – `38615b932cbe` – #15

- Commit: `e89f219` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 3,500 (13.5%)
- Why: feat: elevate all 5 tasks targeting >=0.80 score with DINOv2 and F3 calibration (#15)
- Merged: `d36e163f2f81` | Kaggle public: 0.63286

### 2026-11-07 – `aad484def371` – #14

- Commit: `d96af9d` by Wisit Suwannao
- Prev: `35573ea292e0` | Δrows vs prev: 2,150 (8.3%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore: restore teammate tasks to upstream state post-submission (#14)
- Merged: `1482aa113562` | Kaggle public: –

### 2026-11-06 – `35573ea292e0` – #13

- Commit: `0650975` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,150 (8.3%)
- Why: feat: elevate all 5 tasks (Task 1-5) beating baseline with SOTA models (#13)
- Merged: `8adba6602333` | Kaggle public: 0.63573

### 2026-11-03 – `aad484def371` – #9

- Commit: `f552106` by Wisit Suwannao
- Prev: `39f34e839c43` | Δrows vs prev: 2,330 (9.0%)
- ↩ Same file as 2026-11-02 `e6226cf` (#2)
- Why: chore(tasks-2-4-5): reset secondary tasks to scaffolding for team contributors (#9)
- Merged: `b6a38ac93a15` | Kaggle public: –

### 2026-11-03 – `39f34e839c43` – #7

- Commit: `ba483f9` by Wisit Suwannao
- Prev: `aad484def371` | Δrows vs prev: 2,330 (9.0%)
- Why: feat(tasks-2-4-5): elevate secondary tasks with trained ensemble models and test predictions (#7)
- Merged: `4891205ca6cc` | Kaggle public: 0.58091

### 2026-11-02 – `aad484def371` – #2

- Commit: `e6226cf` by bbb
- Prev: – | Δrows vs prev: new file
- Why: ci: add submission merge check and kaggle submit workflows (#2)
- Merged: `48a77166cd56` | Kaggle public: 0.12007
