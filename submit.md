# Submission Provenance & Workflow (`ml00x` / Team ML004)

> Single source of truth linking every active `task{N}/submit/index.csv` on `main` to the exact code, notebook, and model-level bundle that generated it.

## 1. Current Active `main` Submission Mapping

| Task | Metric | Active `submit/index.csv` Source | Generating Code / Notebook | Required Model Artifacts | Key Validation Score (from code/logs) | `submit/index.csv` SHA-256 (`12c`) | Matches Run 22 (`0.86337`)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1** | $F_2$ | `exp 003` (`bbb`, PR `#68`) | `task1/notebooks/experiments/003.ipynb` | `task1/models/experiments/003/lightgbm.joblib`, `meta.joblib` | `nested_f2=0.83878`, `roc_auc=0.89378` (`15,522` positives / `25,889`) | `fd65d3e13622` | ❌ Differs (`316` rows vs Run 22) — left untouched on `main` |
| **Task 2** | Macro $F_1$ | **Run 22 / Run 18** (`Palapluem`, restored PR `#69`) | `task2/task2_player_segment.ipynb` | `task2/predictions/model-level/task2-exact-param-em-bundle.npz` (+ `Oven_Rev` CSVs) | `OOF ExactBayes Macro-F1=0.75932`, `LogLoss=0.59565` (SLSQP `0.9922 ExactParamEM + 0.0078 LGB_Clean28`) | `d8273939dc94` | ✅ Exact match (`0` diffs) |
| **Task 3** | NMAE | **Run 22 / Run 18** (`Palapluem`, Run 18 `#57`) | `task3/task3_monthly_spending.ipynb` | `task3/predictions/model-level/task3-exact-bayes-v2-bundle.npz` | `OOF NMAE=0.796377`, `MAE=2,683.96 THB` (`ExactBayesL1V2`) | `3646bdd6dda1` | ✅ Exact match (`0` diffs) |
| **Task 4** | Macro $F_1$ | **Run 22** (`Palapluem`, PR `#65`, restored PR `#69`) | `task4/task4_game_title_detection.ipynb` | `task4/predictions/model-level/task4-sgkf-e2e-bundle.npz` + `task4-{eva02,dinov3}-*.csv` (8 files) | `OOF Macro-F1=0.991174`, `LogLoss=0.026605` (9-Stream SGKF incl. `DINOv2-StageC`, Pure-Train Uniform-Prior `alpha*=0.860`) | `1cb773dd067e` | ✅ Exact semantic match (`0` diffs; `index.csv` stores `7-11K..roblock` mapped `1:1` to `0..4` by `submission/merge.py`) |
| **Task 5** | $F_3$ | **Run 22 / Run 18** (`Palapluem`, restored PR `#70`) | `task5/task5_account_security.ipynb` | Self-contained (`task5/data/test.csv` only) | `2,089 / 25,889` flagged (`8.07%`); Unstable-Box `StrictUnflagged_AUC=0.99030`, Archetype 8 (`244`) | `92253faba1f6` | ✅ Exact match (`0` diffs) |

- **Current best Kaggle leaderboard benchmark recorded:** **Run 22 — Public `0.86337`** (used Task 1 Run 19-C `427fb07c946e` + exact Run 22 Tasks 2, 3, 4, 5 above).
- **Merged `submission/submission.csv` hash check:** `b3ea360f52c3` (`uv run submission/merge.py` verified, 25,889 rows, exit 0).

---

## 2. Agreed Team GitHub Workflow (`main`-only + `submit.md`)

1. **No separate throwaway branches / restore PRs:** Develop and land verified updates directly on `main`.
2. **Always keep code + model bundles on `main`:** Any `task{N}/submit/index.csv` update must include the corresponding notebook/script and artifacts in `main`, and update both root `submit.md` and `task{N}/submit/submit.md` in the same commit.
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
