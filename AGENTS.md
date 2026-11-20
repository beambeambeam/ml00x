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
2. From the repository root, run `uv run submission/merge.py --baseline <main checkout>`. It checks all five files, writes `submission/submission.csv` (gitignored), and prints each file's `sha12`, Δrows vs. `main`, and the merged `sha12`.
3. Record the change in the submission history (next section), in the same PR.
4. Open a PR to `main`. The `Check submission` workflow runs the same check and reports what changed vs. `main`. `main` accepts only branches that are up to date with it. When `main` moves, update your branch, rerun `merge.py --baseline`, and fix the history `sha12`, prev, and Δrows values.
5. After merge, start the `Submit to Kaggle` workflow by hand: Actions tab → Submit to Kaggle → Run workflow on `main`. It never runs on push. Each Kaggle note holds a `[sha:<hash>]` tag of its file. If Kaggle already has the same file, the workflow skips the submit. A newer run cancels an older run before it reaches Kaggle, but a submit that has started always finishes. Set `force` to submit a file again. Each submission uses one of the team's daily Kaggle submissions.

Allowed values:

- `task1`, `task5`: `0` or `1`.
- `task2`: `0`–`3`.
- `task3`: THB amount, a number ≥ 0.
- `task4`: `0`–`4` (`0`: 7-11 Knights, `1`: FiveN, `2`: Free Fried, `3`: PUBE, `4`: Roblock per `task4/data/datacard.txt`).

## Submission history

`submit.md` (root) and `task{N}/submit/submit.md` are the submission history. Agents keep them. CI does not check them.

Each time a PR changes `task{N}/submit/index.csv`, including a revert or restore:

1. In `task{N}/submit/submit.md`, add a changelog entry at the top of `## Changelog`. Use the format of the entries below it: date, new `sha12`, PR, commit, author, prev `sha12`, Δrows, why, merged `sha12`. For a revert, add `↩ Same file as <date> <commit>`. Write a real reason in **Why**: run name, notebook, OOF score. Update `## Current` to the new version.
2. In root `submit.md`, add a row at the top of `## Merged history`. Update the `## Current` table and the merged `sha12`.
3. Copy every `sha12` and Δrows value from the `merge.py --baseline` report.

Done when each changed task has a new changelog entry and root `submit.md` has a new merged row. All of them must carry `sha12` values that match the `merge.py` report.

Append only. To fix a wrong entry, add a new entry. Old entries stay as they are.

After a `Submit to Kaggle` run, open a small PR that writes the public score into the matching merged row and changelog entries. Match by merged `sha12`: the run log prints `HASH` and `Public score` (`gh run view <id> --log`).
