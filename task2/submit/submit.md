# Task 2 Current Submission (`submit/index.csv`)

- **Active Version:** **Run 22 / Run 18** (`Palapluem`, restored in PR `#69`)
- **Generating Notebook:** `task2/task2_player_segment.ipynb`
- **Required Artifacts:**
  - `task2/predictions/model-level/task2-exact-param-em-bundle.npz`
  - `task2/predictions/model-level/segment-tuned-catboost-lightgbm-revised-2026-11-12-macro-f1-0.751314-oof.csv`
  - `task2/predictions/model-level/segment-tuned-catboost-lightgbm-revised-2026-11-12-macro-f1-0.751314-test-predictions.csv`
  - `task2/data/train.csv`, `task2/data/test.csv`
- **OOF Metrics:**
  - Nested 5-Fold Outer Stack `ExactBayes` Macro-$F_1$ = **`0.75932`** (`per-fold=[0.75766, 0.7566, 0.76115, 0.75866, 0.76248]`)
  - `OOF LogLoss = 0.59565` (`SLSQP weights: 0.9922 ExactParamEM + 0.0078 LGB_Clean28`)
- **Test Distribution (`25,889` rows):** `0`: `9,455` (`36.5%`), `1`: `6,747` (`26.1%`), `2`: `5,498` (`21.2%`), `3`: `4,189` (`16.2%`)
- **`index.csv` SHA-256:** `d8273939dc9470127d26638bc565b3006f68749156e0e65e8271e3ef803a8aa7`
