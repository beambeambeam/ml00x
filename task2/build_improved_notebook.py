"""Generate the reproducible improvement notebook; run from task2."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python (Task 2)", "language": "python", "name": "task2"}
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown("""# Task 2: tuned gradient boosting

A bounded search to improve the original CatBoost baseline. All selection uses labeled training rows only. We reproduce the original stratified holdout, tune LightGBM with Optuna, and compare candidates using three-fold out-of-fold (OOF) predictions. Both macro and weighted F1 are reported because official averaging has not been confirmed. Macro F1 is the working selection metric, not a claim about Kaggle's scorer.

Holdout tuning results are selection-biased. Three-fold OOF is a stronger diagnostic, but also not a completely untouched nested-CV estimate. Test data is loaded only after model selection. No external data or pretrained models are used.""")
code("""from pathlib import Path
import json
import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from IPython.display import display
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path.cwd()
if not (ROOT / "data/train.csv").exists():
    ROOT = ROOT.parent
for directory in ["models", "predictions", "assets"]:
    (ROOT / directory).mkdir(exist_ok=True)
SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)
train = pd.read_csv(ROOT / "data/train.csv")
y = train["segment"].astype(int)
assert not y.isna().any() and set(y.unique()) == {0, 1, 2, 3}
print("Labeled data:", train.shape)
""")
markdown("""## Stateless feature engineering
Identifiers and documented random noise columns are excluded for the new model. Ratios reflect playtime, social activity, and spending intensity; no aggregate statistics or targets are used. Missing and implausible values are not silently clipped: the tree model can learn their relevance. Categorical vocabularies are learned separately on each training fold, and unseen categories become missing.""")
code("""def features(frame):
    x = frame.drop(columns=["segment", "player_id", "random_metric_1", "random_metric_2", "random_metric_3"], errors="ignore").copy()
    ratios = {
        "spend_per_hour": ("total_spending_thb", "total_playtime_hours"),
        "spend_per_account_day": ("total_spending_thb", "account_age_days"),
        "playtime_per_account_day": ("total_playtime_hours", "account_age_days"),
        "monthly_spend_per_purchase": ("avg_monthly_spending", "spending_frequency"),
        "gifts_per_friend": ("gifts_sent_received", "friend_count"),
        "invites_per_friend": ("friend_invites_sent", "friend_count"),
        "ranked_per_team": ("ranked_participation_rate", "team_play_percentage"),
    }
    for name, (a, b) in ratios.items():
        x[name] = x[a] / (x[b].abs() + 1)
    x["weekly_minutes"] = x["play_frequency"] * x["avg_session_duration"]
    x["social_activity"] = x["friend_count"] * x["team_play_percentage"]
    x["missing_count"] = x.isna().sum(axis=1)
    return x.replace([np.inf, -np.inf], np.nan)

X = features(train)
categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()

def encode_pair(a, b):
    a, b = a.copy(), b.copy()
    for col in categorical:
        dtype = pd.CategoricalDtype(categories=sorted(a[col].dropna().unique()))
        a[col] = a[col].astype(dtype)
        b[col] = b[col].astype(dtype)
    return a, b

indices = np.arange(len(y))
fit_idx, valid_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=SEED)
a, b = encode_pair(X.iloc[fit_idx], X.iloc[valid_idx])

def scores(target, probabilities):
    prediction = probabilities.argmax(axis=1)
    return {"macro_f1": f1_score(target, prediction, average="macro"), "weighted_f1": f1_score(target, prediction, average="weighted")}
""")
code("""def make_model(params, estimators=1500):
    return LGBMClassifier(
        n_estimators=estimators, objective="multiclass", random_state=SEED,
        n_jobs=4, verbosity=-1, **params,
    )

trial_rows = []
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 12, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 250),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
    }
    model = make_model(params)
    model.fit(a, y.iloc[fit_idx], eval_set=[(b, y.iloc[valid_idx])],
              callbacks=[early_stopping(70, verbose=False), log_evaluation(0)])
    result = scores(y.iloc[valid_idx], model.predict_proba(b))
    trial.set_user_attr("iterations", model.best_iteration_)
    trial_rows.append({"trial": trial.number, **result, "iterations": model.best_iteration_, **params})
    print(f"Trial {trial.number}: {result}, iterations={model.best_iteration_}", flush=True)
    return result["macro_f1"]

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=12)
trials = pd.DataFrame(trial_rows).sort_values("macro_f1", ascending=False)
trials.to_csv(ROOT / "assets/segment-tuning-results.csv", index=False)
display(trials.head())
best_params = study.best_params
iterations = max(50, study.best_trial.user_attrs["iterations"])
""")
markdown(
    "## Same-split CatBoost comparison and three-fold OOF evaluation\nReproduce the original baseline rather than comparing only against a remembered score. CV models use fixed iterations, so their held-out fold is never used for early stopping. The final model choice is based on aggregate OOF macro F1."
)
code("""raw = train.drop(columns=["segment", "player_id"]).copy()
raw_categories = raw.select_dtypes(include=["object", "string"]).columns.tolist()
raw[raw_categories] = raw[raw_categories].fillna("__MISSING__").astype(str)

def cat_model():
    return CatBoostClassifier(iterations=300, depth=7, learning_rate=0.08,
        loss_function="MultiClass", random_seed=SEED, thread_count=4,
        allow_writing_files=False, verbose=False)

baseline = cat_model()
baseline.fit(raw.iloc[fit_idx], y.iloc[fit_idx], cat_features=raw_categories)
baseline_holdout = scores(y.iloc[valid_idx], baseline.predict_proba(raw.iloc[valid_idx]))
print("Reproduced baseline holdout:", baseline_holdout)
print("Best tuned holdout:", study.best_value)

oof = {name: np.zeros((len(y), 4)) for name in ["catboost", "lightgbm"]}
cv_rows = []
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for fold, (tr, va) in enumerate(folds.split(X, y), start=1):
    fold_a, fold_b = encode_pair(X.iloc[tr], X.iloc[va])
    lgb = make_model(best_params, iterations)
    lgb.fit(fold_a, y.iloc[tr])
    oof["lightgbm"][va] = lgb.predict_proba(fold_b)
    cat = cat_model()
    cat.fit(raw.iloc[tr], y.iloc[tr], cat_features=raw_categories)
    oof["catboost"][va] = cat.predict_proba(raw.iloc[va])
    for name in oof:
        row = {"fold": fold, "model": name, **scores(y.iloc[va], oof[name][va])}
        cv_rows.append(row)
        print(row, flush=True)

# Fixed equal-weight blending is evaluated without tuning blend weights.
oof["equal_blend"] = (oof["catboost"] + oof["lightgbm"]) / 2
comparison = pd.DataFrame([{"model": name, **scores(y, p)} for name, p in oof.items()]).sort_values("macro_f1", ascending=False)
comparison.to_csv(ROOT / "assets/segment-oof-comparison.csv", index=False)
pd.DataFrame(cv_rows).to_csv(ROOT / "assets/segment-fold-scores.csv", index=False)
winner = comparison.iloc[0]["model"]
display(comparison)
print("Selected:", winner)
print(classification_report(y, oof[winner].argmax(axis=1), digits=4, zero_division=0))
pd.DataFrame(oof[winner], columns=[f"prob_{i}" for i in range(4)]).assign(segment=y.to_numpy()).to_csv(ROOT / "predictions/segment-oof.csv", index=False)
""")
markdown("""## Refit and test predictions
Only now load Test, preserve its row order, and predict with models refitted on **all** labeled rows. Save predictions keyed by the original `player_id`. `ANS` ids are the test row numbers, so each `player_id` maps to its `ANS` id by test row position. Do not overwrite `submit/index.csv` from this notebook; submission changes go through a submission PR.""")
code("""test = pd.read_csv(ROOT / "data/test.csv")
assert "segment" not in test
assert test["player_id"].is_unique
probabilities = []
if winner in ["lightgbm", "equal_blend"]:
    all_x, test_x = encode_pair(X, features(test))
    final_lgb = make_model(best_params, iterations)
    final_lgb.fit(all_x, y)
    probabilities.append(final_lgb.predict_proba(test_x))
    joblib.dump({"model": final_lgb, "categorical_dtypes": {c: all_x[c].dtype for c in categorical}}, ROOT / "models/segment-lightgbm.joblib")
if winner in ["catboost", "equal_blend"]:
    final_cat = cat_model()
    final_cat.fit(raw, y, cat_features=raw_categories)
    test_raw = test[raw.columns].copy()
    test_raw[raw_categories] = test_raw[raw_categories].fillna("__MISSING__").astype(str)
    probabilities.append(final_cat.predict_proba(test_raw))
    final_cat.save_model(str(ROOT / "models/segment-catboost.cbm"))
p = np.mean(probabilities, axis=0)
predictions = pd.DataFrame({"player_id": test["player_id"], "segment": p.argmax(axis=1)})
assert len(predictions) == len(test) and predictions["segment"].isin(range(4)).all()
predictions.to_csv(ROOT / "predictions/segment-best-test.csv", index=False)
metadata = {"selected_model": winner, "seed": SEED, "selection_metric": "macro_f1", "best_params": best_params, "iterations": iterations, "baseline_holdout": baseline_holdout, "oof_scores": comparison.to_dict(orient="records"), "test_rows": len(test)}
(ROOT / "assets/segment-best-result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
display(predictions.head())
print("Saved test predictions:", len(predictions))
print("OOF improvement over baseline:", float(comparison.iloc[0]["macro_f1"] - comparison.set_index("model").loc["catboost", "macro_f1"]))
""")
nb.cells = cells
nbf.write(nb, Path("notebook/player-segment-tuned.ipynb"))
