# Task 4 — Game-title image classification

## Current submission candidate

`submit/index.csv` is built from the **equal-probability soft-voting ensemble of DINOv3 ViT-L, EVA-02 Base, and EVA-02 Large**. It is the current candidate, not a guarantee of the Kaggle score.

| OOF rank | Candidate | Macro F1 | Weighted F1 | Technique |
| ---: | --- | ---: | ---: | --- |
| 1 | DINOv3 ViT-L + EVA-02 Base + EVA-02 Large | **0.991115** | **0.992329** | Equal-weight mean of five-class probabilities; mean fold macro 0.991116, minimum 0.989636 |
| 2 | DINOv3 ViT-L + EVA-02 Large | 0.991047 | 0.992203 | Equal-weight soft vote; mean fold macro 0.991047, minimum 0.990124 |
| 3 | All four models | 0.990530 | 0.991917 | Equal-weight soft vote |
| 4 | All four models | 0.990355 | 0.991790 | Hard vote, soft-probability tie-break |
| 5 | DINOv3 ViT-B + ViT-L + EVA-02 Large | 0.990204 | 0.991568 | Equal-weight soft vote |
| 6 | 25% DINOv3 ViT-L + 75% EVA-02 Large | 0.989305 | 0.990778 | Fixed-weight soft vote |
| 7 | DINOv3 ViT-B + EVA-02 Large | 0.989252 | 0.990715 | Equal-weight soft vote |
| 8 | 25% EVA-02 Base + 75% EVA-02 Large | 0.989142 | 0.990684 | Fixed-weight soft vote |
| 9 | DINOv3 ViT-B + EVA-02 Base + EVA-02 Large | 0.989070 | 0.990651 | Equal-weight soft vote |
| 10 | 25% DINOv3 ViT-B + 75% EVA-02 Large | 0.988892 | 0.990431 | Fixed-weight soft vote |

The top two are effectively near-tied: rank 1 leads by only 0.000068 macro F1, while rank 2 has a slightly better minimum fold and uses fewer models. The scores are five-fold OOF results on the same training labels used to screen 28 predeclared candidates—not independent validation or a public-leaderboard result. Full metrics for all candidates and folds are in [`assets/task4-oof-ensemble-ranking.csv`](assets/task4-oof-ensemble-ranking.csv).

## Notebooks

- [`notebook/eda.ipynb`](notebook/eda.ipynb): full metadata, target-balance, row-order, and optional image EDA. It reads no test labels and saves plots/tables under `assets/eda/`.
- [`notebook/tried/01-eva02-large-dual-t4-oof.ipynb`](notebook/tried/01-eva02-large-dual-t4-oof.ipynb): EVA-02 Large (`eva02_large_patch14_224.mim_m38m`), five-fold OOF and test probabilities. [Kaggle notebook](https://www.kaggle.com/code/microhum/task-4-eva-02-large-dual-t4-5-fold-oof), version 16.
- [`notebook/tried/02-dinov3-vitl-dual-t4-oof.ipynb`](notebook/tried/02-dinov3-vitl-dual-t4-oof.ipynb): DINOv3 ViT-L (`vit_large_patch16_dinov3.lvd1689m`), five-fold OOF and test probabilities. [Kaggle notebook](https://www.kaggle.com/code/microhum/task-4-dinov3-vit-l-dual-t4-5-fold-oof), version 8.
- [`notebook/tried/03-eva02-base-dual-t4-oof.ipynb`](notebook/tried/03-eva02-base-dual-t4-oof.ipynb): EVA-02 Base (`eva02_base_patch14_224.mim_in22k`), local notebook source using the EVA-02 pipeline.
- [`notebook/tried/04-dinov3-vitb-dual-t4-oof.ipynb`](notebook/tried/04-dinov3-vitb-dual-t4-oof.ipynb): DINOv3 ViT-B (`vit_base_patch16_dinov3.lvd1689m`), local notebook source using the DINOv3 pipeline.

The two published Kaggle links are the Large-model runs. Base variants are included locally but are not represented as separately published Kaggle runs. The training notebooks use the Kaggle image dataset `microhum/cpe333-task-4-dataset`; the local `data/` directory contains the supplied CSV metadata and datacard, not the large image folders. All four notebook sources now save one best-epoch inference checkpoint per fold and bundle them with their run artifacts. The published Kaggle versions linked above are the earlier executed versions; these local source updates have not yet been run or published on Kaggle. The previously downloaded runs supplied probability CSVs but no checkpoint files, so the existing checkpoints cannot be recovered from those CSVs; rerun a notebook on Kaggle and save/download its artifacts ZIP to obtain the checkpoints.

## Included predictions and submission mapping

All inputs required to reproduce the local ensemble are in this task folder:

- `predictions/model-level/`: OOF and test-probability CSVs for DINOv3 ViT-B/ViT-L and EVA-02 Base/Large.
- `predictions/task4-f1-0p9911-soft-mean-dinov3-vitl-eva02-base-eva02-large-oof-predictions.csv`: the selected ensemble's OOF probabilities, labels, and folds.
- `predictions/task4-f1-0p9911-soft-mean-dinov3-vitl-eva02-base-eva02-large-test-predictions.csv`: selected five-class probabilities and labels for the test rows.
- `assets/task4-oof-ensemble-ranking.csv`: all 28 OOF candidates, sorted by macro F1.
- `submit/index.csv`: current rank-1 candidate in the required `id,task4` format.

The prediction export carries IDs `ANS00001` through `ANS25889`, which encode the test-row positions. The index builder checks the prediction IDs against the shared `submission/sample_submission.csv` order and checks the Task 4 test row count, then writes the labels in that verified order. It does not infer an ID from a filename or use any test labels. Class mapping: `0 → 7-11K`, `1 → fiveN`, `2 → freefried`, `3 → pube`, `4 → roblock`.

## Reproduce and verify

Run from `task4/`:

```powershell
uv run python scripts/ensemble_oof_predictions.py
uv run python scripts/build_task4_index.py
uv run python scripts/verify_task4_artifacts.py
```

The ensemble script reads the included model-level CSVs by default, validates their OOF/test alignment, evaluates the fixed candidate set, and writes the full ranking plus the selected test predictions. To execute EDA locally:

```powershell
uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace notebook/eda.ipynb
```

From the repository root, validate the combined five-task submission:

```powershell
uv run submission/merge.py
```

The image-only EDA sections are skipped when image folders are not present locally; the Kaggle training notebooks mount them from the dataset. OOF performance is a screening estimate, and the small top-two difference should not be treated as statistically established.
