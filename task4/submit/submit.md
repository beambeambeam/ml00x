# Task 4 Current Submission (`submit/index.csv`)

- **Active Version:** **Run 22** (`Palapluem`, PR `#65`, restored in PR `#69`)
- **Generating Notebook:** `task4/task4_game_title_detection.ipynb`
- **Required Artifacts:**
  - `task4/predictions/model-level/task4-sgkf-e2e-bundle.npz` (5-fold SGKF Pure-Train E2E `ConvNeXtV2`, `DINOv2-StageA/C`, `MLP-V3`)
  - 8 model-level CSVs in `task4/predictions/model-level/`: `task4-{dinov3-vitb,dinov3-vitl,eva02-base,eva02-large}-{oof,test-predictions}.csv`
  - `task4/data/train.csv`, `task4/data/test.csv`
- **Method & OOF Metrics (`31,546` train images, 100% Pure-Train):**
  - Temperature-calibrated 9-stream Log-Odds SLSQP pool (`EVA02-Large=0.3158`, `DINOv3-ViT-L=0.2017`, `ConvNeXtV2-SGKF=0.1867`, `DINOv2-StageC-2TTA=0.1402`, `EVA02-Base=0.0772`, `DINOv2-StageC-6TTA=0.0627`, `MLP-V3-SGKF=0.0143`, `DINOv3-ViT-B=0.0013`) + Pure-Train Nested Uniform-Prior Debias (`alpha*=0.860`)
  - `OOF Macro-F1 = 0.991174` | `OOF LogLoss = 0.026605` | `Iso(<0.75) F1 = 0.991955` | `Corrupt F1 = 0.986183`
- **Test Distribution (`25,889` rows):** `0` (`7-11K`): `3,667` (`14.2%`), `1` (`fiveN`): `6,832` (`26.4%`), `2` (`freefried`): `3,865` (`14.9%`), `3` (`pube`): `6,407` (`24.7%`), `4` (`roblock`): `5,118` (`19.8%`)
  - *Note:* `submit/index.csv` currently retains string class names (`7-11K..roblock`, SHA-256 `1cb773dd...`), which `submission/merge.py` maps 1:1 to `0..4` identically to integer outputs from `task4_game_title_detection.ipynb`.
- **`index.csv` SHA-256:** `1cb773dd067e3c7bb07e118e9218bb3a473cbfe39dfe49cb4624ee0ff622b27c`
