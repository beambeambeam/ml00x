"""Replay saved Task 2 OOF models and fix the validation/export rule mismatch.

Run from task2/: uv run python run_bayes_export.py [--promote]
No base models are retrained. Test rows only receive predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from bayes_decision import (
    fit_exact_bayes_macro_f1,
    fit_slsqp_weights,
    predict_segments,
    validate_probabilities,
)

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = [
    "ExactParamEM",
    "XGB_Clean28",
    "LGB_Clean28",
    "CB_Clean28",
    "CleanTabM_SWA3",
    "Fact_Clean28",
    "Oven_Rev",
]
BUNDLE = ROOT / "predictions/model-level/task2-exact-param-em-bundle.npz"
OVEN_PREFIX = "segment-tuned-catboost-lightgbm-revised-2026-11-12-macro-f1-0.751314"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(y, prediction):
    matrix = np.bincount(4 * y + prediction, minlength=16).reshape(4, 4)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.mean(2 * matrix.diagonal() / np.maximum(denominator, 1)))


def load_probabilities(train, test):
    paths = [BUNDLE]
    with np.load(BUNDLE) as bundle:
        keys = ["em", "xgb_c", "lgb_c", "cb_c", "tabm_c", "fact"]
        oof = [bundle[f"oof_{key}"] for key in keys]
        inference = [bundle[f"te_{key}"] for key in keys]
    for suffix, frame, arrays in (
        ("oof", train, oof),
        ("test-predictions", test, inference),
    ):
        path = BUNDLE.parent / f"{OVEN_PREFIX}-{suffix}.csv"
        saved = pd.read_csv(path)
        if not saved.player_id.is_unique or set(saved.player_id) != set(frame.player_id):
            raise ValueError(f"Saved model IDs do not match data: {path}")
        aligned = saved.set_index("player_id").loc[frame.player_id]
        if suffix == "oof" and not np.array_equal(aligned.segment.to_numpy(), train.segment):
            raise ValueError("Saved OOF labels do not match train.csv")
        arrays.append(aligned[[f"blend_prob_{c}" for c in range(4)]].to_numpy())
        paths.append(path)
    for arrays, rows in ((oof, len(train)), (inference, len(test))):
        for i, array in enumerate(arrays):
            # Match the frozen notebook's normalization before fitting the stack.
            array = np.clip(np.asarray(array, dtype=float), 1e-12, 1.0)
            array /= array.sum(axis=1, keepdims=True)
            arrays[i] = validate_probabilities(array)
            if array.shape != (rows, 4):
                raise ValueError("Frozen bundle row count does not match data")
    return oof, inference, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="Write corrected submit/index.csv")
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    if not train.player_id.is_unique or not test.player_id.is_unique:
        raise ValueError("Data must have unique player IDs")
    if "segment" in test or set(train.segment.unique()) != {0, 1, 2, 3}:
        raise ValueError("Expected labeled train data and unlabeled test data")
    y = train.segment.to_numpy(dtype=int)
    oof, inference, inputs = load_probabilities(train, test)
    probabilities = np.zeros_like(oof[0])
    old_prediction, prediction, folds = (np.empty(len(y), dtype=int) for _ in range(3))
    fold_results, fold_parameters = [], []
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        weights = fit_slsqp_weights([p[tr] for p in oof], y[tr])
        train_prob = sum(w * p[tr] for w, p in zip(weights, oof, strict=True))
        valid_prob = sum(w * p[va] for w, p in zip(weights, oof, strict=True))
        alpha, beta, _ = fit_exact_bayes_macro_f1(train_prob)
        probabilities[va] = valid_prob
        # Reproduce the obsolete export for comparison only.
        old_prediction[va] = (valid_prob * alpha).argmax(axis=1)
        prediction[va] = predict_segments(valid_prob, alpha, beta)
        folds[va] = fold
        old_score, fixed_score = (
            macro_f1(y[va], old_prediction[va]),
            macro_f1(y[va], prediction[va]),
        )
        fold_results.append(
            {
                "fold": fold,
                "old_export_macro_f1": old_score,
                "fixed_macro_f1": fixed_score,
                "delta": fixed_score - old_score,
            }
        )
        fold_parameters.append(
            {
                "fold": fold,
                "weights": weights.tolist(),
                "alpha": alpha.tolist(),
                "beta": beta.tolist(),
            }
        )
        print(f"Fold {fold}: {old_score:.6f} -> {fixed_score:.6f}", flush=True)

    old_score, fixed_score = macro_f1(y, old_prediction), macro_f1(y, prediction)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(1000):
        indices = rng.integers(0, len(y), len(y))
        deltas.append(
            macro_f1(y[indices], prediction[indices])
            - macro_f1(y[indices], old_prediction[indices])
        )
    interval = np.quantile(deltas, [0.025, 0.975]).tolist()
    gate = all(row["delta"] > 0 for row in fold_results) and interval[0] > 0
    if args.promote and not gate:
        raise ValueError("Promotion requires gains on every fold and a positive bootstrap interval")

    # Freeze all decisions before using test probabilities.
    weights = fit_slsqp_weights(oof, y)
    alpha, beta, _ = fit_exact_bayes_macro_f1(probabilities)
    test_prob = sum(w * p for w, p in zip(weights, inference, strict=True))
    fixed_test = predict_segments(test_prob, alpha, beta)
    old_test = (test_prob * alpha).argmax(axis=1)
    sample_path = ROOT.parent / "submission/sample_submission.csv"
    sample = pd.read_csv(sample_path, usecols=["id"])
    if len(sample) != len(test) or not sample.id.is_unique or sample.id.isna().any():
        raise ValueError("Sample submission does not align with test rows")
    # Preserve the active notebook's positional test-to-ANS mapping.
    submission = sample.assign(task2=fixed_test)
    current = pd.read_csv(ROOT / "submit/index.csv")
    if not current.id.equals(sample.id):
        raise ValueError("Current submission ID order differs from the official sample")
    if not (np.array_equal(current.task2, old_test) or np.array_equal(current.task2, fixed_test)):
        raise ValueError("Current submission matches neither the frozen old nor fixed export")

    assets, models = ROOT / "assets/bayes-export", ROOT / "models/bayes-export"
    outputs = ROOT / "predictions/bayes-export"
    for directory in (assets, models, outputs):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(assets / "per-fold-macro-f1.csv", index=False)
    pd.DataFrame(
        {
            "player_id": train.player_id,
            "fold": folds,
            "segment": y,
            **{f"prob_{c}": probabilities[:, c] for c in range(4)},
            "old_prediction": old_prediction,
            "prediction": prediction,
        }
    ).to_csv(outputs / "oof.csv", index=False)
    full_test = pd.DataFrame(
        {
            "id": sample.id,
            "player_id": test.player_id,
            **{f"prob_segment_{c}": test_prob[:, c] for c in range(4)},
            "old_prediction": old_test,
            "segment": fixed_test,
        }
    )
    full_test.to_csv(outputs / "test.csv", index=False)
    submission.to_csv(outputs / "index.csv", index=False)
    metadata = {
        "method": "Existing seven-model stack with shared affine macro-F1 decision rule",
        "protocol": "Five-fold cross-fitted postprocessing of frozen OOF probabilities; no base-model retraining",
        "limitations": "Not fully nested retraining: other OOF base models can have seen held-out fold labels. Bootstrap is conditional on frozen models and folds. NPZ has no row IDs; original row order is assumed and input hashes are recorded.",
        "model_names": MODEL_NAMES,
        "weights": weights.tolist(),
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "fold_parameters": fold_parameters,
        "old_export_macro_f1": old_score,
        "fixed_macro_f1": fixed_score,
        "fixed_weighted_f1": float(f1_score(y, prediction, average="weighted")),
        "old_export_weighted_f1": float(f1_score(y, old_prediction, average="weighted")),
        "oof_log_loss": float(log_loss(y, probabilities)),
        "bootstrap_delta_macro_f1_95_percent": interval,
        "passed_promotion_gate": gate,
        "test_changed_rows": int((old_test != fixed_test).sum()),
        "test_class_counts": np.bincount(fixed_test, minlength=4).tolist(),
        "input_sha256": {
            str(p.relative_to(ROOT.parent)): sha256(p)
            for p in [ROOT / "data/train.csv", ROOT / "data/test.csv", sample_path, *inputs]
        },
    }
    (models / "decision-rule.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(5)
    ax.bar(
        positions - 0.18, [r["old_export_macro_f1"] for r in fold_results], 0.36, label="Old export"
    )
    ax.bar(
        positions + 0.18,
        [r["fixed_macro_f1"] for r in fold_results],
        0.36,
        label="Fixed affine rule",
    )
    ax.set(
        xticks=positions,
        xticklabels=range(1, 6),
        xlabel="Fold",
        ylabel="Macro F1",
        ylim=(0.74, 0.77),
        title="Task 2: validation and export use the same rule",
    )
    ax.legend()
    fig.savefig(assets / "export-rule-macro-f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if args.promote:
        submission.to_csv(ROOT / "submit/index.csv", index=False)
        full_test.drop(columns="old_prediction").to_csv(
            ROOT / "predictions/test_predictions.csv", index=False
        )
    print(f"OOF macro F1: {old_score:.6f} -> {fixed_score:.6f}; gain CI {interval}")
    print(f"Test changes: {metadata['test_changed_rows']}; promotion gate: {gate}")


if __name__ == "__main__":
    main()
