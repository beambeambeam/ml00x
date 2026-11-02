# AGENTS.md

## Project brief

Read `docs/project-instruction.md` for background on each task: situation, example features, and hints. Each task has its own Kaggle competition page. If the two differ, the Kaggle page is correct.

| Task | Type                       | Target         | Metric   | Pre-trained models |
| ---- | -------------------------- | -------------- | -------- | ------------------ |
| 1    | Binary classification      | `is_cheater`   | F2       | Not allowed        |
| 2    | Multi-class classification | `segment`      | F1       | Not allowed        |
| 3    | Regression                 | `spending_30d` | NMAE     | Not allowed        |
| 4    | Image classification       | `game_title`   | Macro F1 | Allowed            |
| 5    | Anomaly detection          | `is_anomaly`   | F3       | Not allowed        |

## Repository layout

Each task is an independent uv project in its own folder.

```
docs/        project instruction
submission/  sample submission and merge script
task1/       Task 1 uv project
task2/       Task 2 uv project
task3/       Task 3 uv project
task4/       Task 4 uv project
task5/       Task 5 uv project
```

Each task folder holds:

- `pyproject.toml`, `uv.lock`, `.python-version`: the task's dependencies and Python version. Commit all three.
- `.venv/`: the task's virtual environment. uv creates it.
- `.gitignore`: the task's ignore rules. Ignore `.venv/`. Do not ignore `data/`, `models/`, or `predictions/`.
- `data/`: the task's Kaggle files. Download them here and commit them.
- `models/`: saved model checkpoints. Commit them.
- `predictions/`: prediction files from experiments. Commit them.
- `submit/index.csv`: the one prediction file this task sends to Kaggle. Columns `id,task{N}`, all 25,889 ids. Commit it.
- `assets/`: report figures and tables that notebooks and scripts generate. Commit this folder.

Rules for report assets:

- Save each figure from code into the task's `assets/` folder. The code that makes a figure stays in the task folder, so the team can make it again.
- Name each file in kebab-case by its content, for example `confusion-matrix.png` or `feature-importance.png`.
- Save PNG figures at 300 dpi with `bbox_inches="tight"`.
- Save report tables as CSV or Markdown.

Rules for task folders:

- Run every command from inside the task folder. Use `uv run <command>` so the command uses that task's `.venv`.
- Add packages with `uv add <package>` inside the task folder. Each task keeps its own dependency set.
- Keep each task self-contained. A task folder imports no code from another task folder.

## Competition rules

These rules are hard guardrails. A violation can disqualify the team.

- Use only the provided Kaggle datasets.
- Train on Train and Val combined.
- Use the Test set only to make predictions. Test rows never enter model fitting, pseudo-labeling, or fine-tuning.
- Task 5 is the exception: it has only `test.csv`, so the unsupervised model fits on it without labels.
- Train Tasks 1, 2, 3, and 5 from scratch. Use pre-trained weights only in Task 4.
- Use hyperparameter tuning tools such as Optuna or scikit-learn search. Never use AutoML frameworks: AutoGluon, auto-sklearn, H2O AutoML, FLAML, TPOT, PyCaret.
- Write all code inside the team. Share no code or predictions with other teams.

## Metric-driven choices

- Tasks 1 and 5 weight recall above precision (F2, F3). Tune each decision threshold on out-of-fold predictions to maximize that task's metric.
- Task 3 data is zero-inflated. If you train on a log target, invert the transform before you write predictions in THB.
- Use out-of-fold cross-validation scores to compare models. Use the public leaderboard as a second check only.

## Submission

One Kaggle competition takes one CSV with all five tasks. CI builds it from the five `task{N}/submit/index.csv` files.

To update a task's submission:

1. Write `task{N}/submit/index.csv` with columns `id,task{N}`. Use every id in `submission/sample_submission.csv`.
2. From the repository root, run `uv run submission/merge.py`. It checks all five files and writes `submission/submission.csv` (gitignored).
3. Open a PR to `main`. The `Check submission` workflow runs the same check and reports what changed vs. `main`.
4. After merge, the `Submit to Kaggle` workflow runs on `main` if the push changes `submission/` or a `task{N}/submit/` file. A newer run cancels an older run that is still in progress. You can also start it from the Actions tab. Each submission uses one of the team's daily Kaggle submissions.

Allowed values:

- `task1`, `task5`: `0` or `1`.
- `task2`: `0`–`3`.
- `task3`: THB amount, a number ≥ 0.
- `task4`: one of `7-11K`, `fiveN`, `freefried`, `pube`, `roblock`.
