from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASS_NAMES = ["7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock"]
MODEL_FILES = {
    "dinov3-vitb": "task4-dinov3-vitb",
    "dinov3-vitl": "task4-dinov3-vitl",
    "eva02-base": "task4-eva02-base",
    "eva02-large": "task4-eva02-large",
}
PROB_COLS = [f"prob_{i}" for i in range(len(CLASS_NAMES))]
BASE = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "predictions" / "model-level"
)
ASSETS_DIR = BASE / "assets"
PREDICTIONS_DIR = BASE / "predictions"


def read_predictions() -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    oof_frames = {}
    test_frames = {}
    oof_probabilities = {}
    test_probabilities = {}

    for model_name, file_stem in MODEL_FILES.items():
        oof = pd.read_csv(INPUT_DIR / f"{file_stem}-oof.csv")
        test = pd.read_csv(INPUT_DIR / f"{file_stem}-test-predictions.csv")
        required_oof = {"id", "file_name", "label", "fold", *PROB_COLS}
        required_test = {
            "id",
            "file_name",
            "predicted_label",
            "predicted_class",
            *PROB_COLS,
        }
        assert required_oof.issubset(oof.columns), f"{model_name}: OOF columns missing"
        assert required_test.issubset(test.columns), (
            f"{model_name}: test columns missing"
        )
        assert "label" not in test.columns, (
            f"{model_name}: test labels must not be present"
        )
        assert oof["id"].is_unique and test["id"].is_unique
        assert oof["file_name"].is_unique and test["file_name"].is_unique

        oof_frames[model_name] = oof
        test_frames[model_name] = test
        oof_probabilities[model_name] = oof[PROB_COLS].to_numpy(dtype=np.float64)
        test_probabilities[model_name] = test[PROB_COLS].to_numpy(dtype=np.float64)

    reference = oof_frames[next(iter(MODEL_FILES))]
    reference_test = test_frames[next(iter(MODEL_FILES))]
    for model_name in MODEL_FILES:
        oof = oof_frames[model_name]
        test = test_frames[model_name]
        for column in ("id", "file_name", "label", "fold"):
            assert oof[column].equals(reference[column]), (
                f"{model_name}: OOF {column} misaligned"
            )
        for column in ("id", "file_name"):
            assert test[column].equals(reference_test[column]), (
                f"{model_name}: test {column} misaligned"
            )

        p_oof = oof_probabilities[model_name]
        p_test = test_probabilities[model_name]
        assert np.isfinite(p_oof).all() and (p_oof >= 0).all()
        assert np.isfinite(p_test).all() and (p_test >= 0).all()
        assert np.allclose(p_oof.sum(axis=1), 1.0, atol=1e-3)
        assert np.allclose(p_test.sum(axis=1), 1.0, atol=1e-3)
        assert np.array_equal(test["predicted_label"].to_numpy(), p_test.argmax(axis=1))

    labels = reference["label"].to_numpy(dtype=np.int64)
    folds = reference["fold"].to_numpy(dtype=np.int64)
    assert set(np.unique(labels)) == set(range(len(CLASS_NAMES)))
    assert len(np.unique(folds)) == 5, (
        f"Expected five OOF folds; got {np.unique(folds)}"
    )
    return reference, reference_test, oof_probabilities, test_probabilities


def main() -> None:
    reference, reference_test, oof_probs, test_probs = read_predictions()
    labels = reference["label"].to_numpy(dtype=np.int64)
    folds = reference["fold"].to_numpy(dtype=np.int64)
    candidate_rows: list[dict] = []
    oof_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add_candidate(
        name: str,
        method: str,
        source_models: tuple[str, ...],
        oof_probability: np.ndarray,
        test_probability: np.ndarray,
        prediction: np.ndarray | None = None,
        test_prediction: np.ndarray | None = None,
    ) -> None:
        oof_prediction = (
            oof_probability.argmax(axis=1) if prediction is None else prediction
        )
        test_prediction = (
            test_probability.argmax(axis=1)
            if test_prediction is None
            else test_prediction
        )
        fold_scores = [
            f1_score(
                labels[folds == fold],
                oof_prediction[folds == fold],
                labels=range(5),
                average="macro",
            )
            for fold in sorted(np.unique(folds))
        ]
        row = {
            "method": name,
            "ensemble_type": method,
            "models": "+".join(source_models),
            "model_count": len(source_models),
            "oof_macro_f1": f1_score(
                labels, oof_prediction, labels=range(5), average="macro"
            ),
            "oof_weighted_f1": f1_score(
                labels, oof_prediction, labels=range(5), average="weighted"
            ),
            "mean_fold_macro_f1": float(np.mean(fold_scores)),
            "min_fold_macro_f1": float(np.min(fold_scores)),
            "fold_macro_f1": ";".join(f"{score:.6f}" for score in fold_scores),
        }
        candidate_rows.append(row)
        oof_outputs[name] = (oof_probability, oof_prediction)
        test_outputs[name] = (test_probability, test_prediction)

    # Individual OOF model scores.
    for model_name in MODEL_FILES:
        add_candidate(
            model_name,
            "single_model",
            (model_name,),
            oof_probs[model_name],
            test_probs[model_name],
        )

    # Fixed, equal-weight soft voting across every model subset.
    for size in range(2, len(MODEL_FILES) + 1):
        for subset in combinations(MODEL_FILES, size):
            oof_mean = np.mean([oof_probs[name] for name in subset], axis=0)
            test_mean = np.mean([test_probs[name] for name in subset], axis=0)
            label = "soft-mean-" + "-".join(subset)
            add_candidate(label, "equal_weight_soft_vote", subset, oof_mean, test_mean)

    # Two predeclared asymmetric pair blends, to check whether the less accurate
    # model adds useful diversity without fitting an ensemble weight to OOF labels.
    for left, right in combinations(MODEL_FILES, 2):
        for left_weight in (0.25, 0.75):
            right_weight = 1.0 - left_weight
            oof_blend = left_weight * oof_probs[left] + right_weight * oof_probs[right]
            test_blend = (
                left_weight * test_probs[left] + right_weight * test_probs[right]
            )
            name = f"soft-{left_weight:.0%}-{left}-{right_weight:.0%}-{right}"
            add_candidate(
                name, "fixed_weight_soft_vote", (left, right), oof_blend, test_blend
            )

    # Hard vote for all four, using mean class probability only to break ties.
    model_names = tuple(MODEL_FILES)
    oof_stack = np.stack([oof_probs[name] for name in model_names])
    test_stack = np.stack([test_probs[name] for name in model_names])
    oof_votes = np.eye(5, dtype=np.int8)[oof_stack.argmax(axis=2)].sum(axis=0)
    test_votes = np.eye(5, dtype=np.int8)[test_stack.argmax(axis=2)].sum(axis=0)
    oof_tiebreak = np.mean(oof_stack, axis=0)
    test_tiebreak = np.mean(test_stack, axis=0)
    oof_hard = np.where(
        oof_votes == oof_votes.max(axis=1, keepdims=True), oof_votes, -1
    )
    test_hard = np.where(
        test_votes == test_votes.max(axis=1, keepdims=True), test_votes, -1
    )
    oof_hard_pred = np.where(
        oof_hard.max(axis=1, keepdims=True) == oof_hard, oof_tiebreak, -1
    ).argmax(axis=1)
    test_hard_pred = np.where(
        test_hard.max(axis=1, keepdims=True) == test_hard, test_tiebreak, -1
    ).argmax(axis=1)
    oof_hard_probabilities = np.eye(5, dtype=np.float64)[oof_hard_pred]
    test_hard_probabilities = np.eye(5, dtype=np.float64)[test_hard_pred]
    add_candidate(
        "hard-vote-all-four-soft-probability-tiebreak",
        "majority_hard_vote",
        model_names,
        oof_hard_probabilities,
        test_hard_probabilities,
        prediction=oof_hard_pred,
        test_prediction=test_hard_pred,
    )

    ranking = pd.DataFrame(candidate_rows).sort_values(
        ["oof_macro_f1", "oof_weighted_f1", "min_fold_macro_f1"],
        ascending=False,
        ignore_index=True,
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    top10 = ranking.head(10).copy()
    best = ranking.iloc[0]
    best_oof_probabilities, best_oof_labels = oof_outputs[best["method"]]
    best_test_probabilities, best_test_labels = test_outputs[best["method"]]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = ASSETS_DIR / "task4-oof-ensemble-ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    method_slug = re.sub(r"[^a-z0-9]+", "-", str(best["method"]).lower()).strip("-")[
        :80
    ]
    score_slug = f"{float(best['oof_macro_f1']):.4f}".replace(".", "p")
    oof_path = (
        PREDICTIONS_DIR / f"task4-f1-{score_slug}-{method_slug}-oof-predictions.csv"
    )
    oof_result = reference[["id", "file_name", "label", "fold"]].copy()
    oof_result["predicted_label"] = best_oof_labels.astype(int)
    oof_result["predicted_class"] = [CLASS_NAMES[label] for label in best_oof_labels]
    for class_idx, column in enumerate(PROB_COLS):
        oof_result[column] = best_oof_probabilities[:, class_idx]
    assert len(oof_result) == len(reference)
    assert oof_result["id"].equals(reference["id"])
    assert np.array_equal(
        oof_result["predicted_label"], best_oof_probabilities.argmax(axis=1)
    )
    oof_result.to_csv(oof_path, index=False)

    test_path = (
        PREDICTIONS_DIR / f"task4-f1-{score_slug}-{method_slug}-test-predictions.csv"
    )
    test_result = reference_test[["id", "file_name"]].copy()
    test_result["predicted_label"] = best_test_labels.astype(int)
    test_result["predicted_class"] = [CLASS_NAMES[label] for label in best_test_labels]
    for class_idx, column in enumerate(PROB_COLS):
        test_result[column] = best_test_probabilities[:, class_idx]
    assert len(test_result) == len(reference_test)
    assert test_result["id"].equals(reference_test["id"])
    assert test_result["id"].is_unique
    assert "label" not in test_result.columns
    saved_probabilities = test_result[PROB_COLS].to_numpy(dtype=np.float64)
    assert np.isfinite(saved_probabilities).all() and (saved_probabilities >= 0).all()
    assert np.allclose(saved_probabilities.sum(axis=1), 1.0, atol=1e-3)
    assert np.array_equal(
        test_result["predicted_label"], saved_probabilities.argmax(axis=1)
    )
    test_result.to_csv(test_path, index=False)

    print(
        f"Aligned OOF rows: {len(reference):,}; test rows: {len(reference_test):,}; candidates: {len(ranking)}"
    )
    print(f"OOF fold ids: {sorted(np.unique(folds).tolist())}; test labels loaded: no")
    single_models = ranking[ranking["ensemble_type"] == "single_model"]
    print("Individual model OOF scores:")
    print(
        single_models[
            ["method", "oof_macro_f1", "oof_weighted_f1", "mean_fold_macro_f1"]
        ].to_string(index=False)
    )
    print("Top 10 candidates:")
    print(
        top10[
            [
                "rank",
                "method",
                "oof_macro_f1",
                "oof_weighted_f1",
                "mean_fold_macro_f1",
                "min_fold_macro_f1",
            ]
        ].to_string(index=False)
    )
    print(f"Ranking: {ranking_path}")
    print(f"Best OOF predictions: {oof_path}")
    print(f"Best test predictions: {test_path}")


if __name__ == "__main__":
    main()
