from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parent
MODEL_DIR = TASK_DIR / "predictions" / "model-level"
SUBMISSION_LABELS = ["7-11K", "fiveN", "freefried", "pube", "roblock"]
CLASS_TO_INDEX = {
    "7-11k": 0,
    "fiven": 1,
    "freefried": 2,
    "pube": 3,
    "roblock": 4,
}
DISPLAY_NAMES = ["7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock"]
MODEL_STEMS = [
    "task4-dinov3-vitb",
    "task4-dinov3-vitl",
    "task4-eva02-base",
    "task4-eva02-large",
]
PROB_COLS = [f"prob_{i}" for i in range(5)]
BEST_METHOD_STEM = "task4-f1-0p9911-soft-mean-dinov3-vitl-eva02-base-eva02-large"
BEST_OOF_PREDICTIONS = (
    TASK_DIR / "predictions" / f"{BEST_METHOD_STEM}-oof-predictions.csv"
)
BEST_PREDICTIONS = TASK_DIR / "predictions" / f"{BEST_METHOD_STEM}-test-predictions.csv"


def check_probabilities(frame: pd.DataFrame, name: str) -> np.ndarray:
    probabilities = frame[PROB_COLS].to_numpy(dtype=np.float64)
    assert np.isfinite(probabilities).all(), f"{name}: non-finite probabilities"
    assert (probabilities >= 0).all(), f"{name}: negative probabilities"
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3), (
        f"{name}: probabilities do not sum to one"
    )
    return probabilities


def main() -> None:
    train = pd.read_csv(TASK_DIR / "data" / "train.csv", dtype={"id": str})
    test = pd.read_csv(
        TASK_DIR / "data" / "test.csv", usecols=["id"], dtype={"id": str}
    )
    sample = pd.read_csv(
        REPO_ROOT / "submission" / "sample_submission.csv",
        usecols=["id"],
        dtype={"id": str},
    )
    assert len(test) == len(sample), "Task 4 test/sample row counts differ"
    expected_sample_ids = pd.Series(
        [f"ANS{i:05d}" for i in range(1, len(sample) + 1)], name="id"
    )
    assert sample["id"].reset_index(drop=True).equals(expected_sample_ids), (
        "Sample IDs are not sequential ANS test-row IDs"
    )

    expected_labels = (
        train["class"].astype(str).str.strip().str.lower().map(CLASS_TO_INDEX)
    )
    assert expected_labels.notna().all(), "Unknown or missing training class"
    expected_filenames = train["id"].astype(str) + ".jpg"
    reference_oof: pd.DataFrame | None = None
    reference_test: pd.DataFrame | None = None
    for stem in MODEL_STEMS:
        oof = pd.read_csv(MODEL_DIR / f"{stem}-oof.csv", dtype={"file_name": str})
        test_predictions = pd.read_csv(
            MODEL_DIR / f"{stem}-test-predictions.csv", dtype={"id": str}
        )
        required_oof = {"id", "file_name", "label", "fold", *PROB_COLS}
        required_test = {
            "id",
            "file_name",
            "predicted_label",
            "predicted_class",
            *PROB_COLS,
        }
        assert required_oof.issubset(oof.columns), f"{stem}: OOF columns missing"
        assert required_test.issubset(test_predictions.columns), (
            f"{stem}: test columns missing"
        )
        assert "label" not in test_predictions.columns, (
            f"{stem}: test labels must not be included"
        )
        assert len(oof) == len(train), f"{stem}: OOF/train row count mismatch"
        assert (
            oof["file_name"]
            .astype(str)
            .reset_index(drop=True)
            .equals(expected_filenames.reset_index(drop=True))
        ), f"{stem}: OOF rows do not align to train.csv order"
        assert np.array_equal(
            pd.to_numeric(oof["label"], errors="raise").to_numpy(dtype=int),
            expected_labels.to_numpy(dtype=int),
        ), f"{stem}: OOF labels do not align with train.csv"
        assert set(pd.to_numeric(oof["fold"], errors="raise").unique()) == set(
            range(5)
        ), f"{stem}: expected five folds"
        assert oof["id"].is_unique and oof["file_name"].is_unique
        assert (
            test_predictions["id"].reset_index(drop=True).equals(expected_sample_ids)
        ), f"{stem}: test predictions are not in ANS row-index order"
        assert test_predictions["id"].is_unique
        check_probabilities(oof, f"{stem} OOF")
        test_probabilities = check_probabilities(test_predictions, f"{stem} test")
        assert np.array_equal(
            pd.to_numeric(test_predictions["predicted_label"], errors="raise").to_numpy(
                dtype=int
            ),
            test_probabilities.argmax(axis=1),
        ), f"{stem}: test labels do not match probability argmax"

        if reference_oof is None:
            reference_oof = oof
            reference_test = test_predictions
        else:
            for column in ("id", "file_name", "label", "fold"):
                assert oof[column].equals(reference_oof[column]), (
                    f"{stem}: OOF {column} misaligned across models"
                )
            for column in ("id", "file_name"):
                assert test_predictions[column].equals(reference_test[column]), (
                    f"{stem}: test {column} misaligned across models"
                )

    selected_oof = pd.read_csv(BEST_OOF_PREDICTIONS, dtype={"file_name": str})
    assert reference_oof is not None
    assert len(selected_oof) == len(train)
    for column in ("id", "file_name", "label", "fold"):
        assert selected_oof[column].equals(reference_oof[column]), (
            f"Selected ensemble OOF {column} misaligned"
        )
    oof_probabilities = check_probabilities(selected_oof, "selected ensemble OOF")
    oof_labels = pd.to_numeric(
        selected_oof["predicted_label"], errors="raise"
    ).to_numpy(dtype=int)
    assert np.array_equal(oof_labels, oof_probabilities.argmax(axis=1))
    assert np.array_equal(
        pd.to_numeric(selected_oof["label"], errors="raise").to_numpy(dtype=int),
        expected_labels.to_numpy(dtype=int),
    )

    selected = pd.read_csv(BEST_PREDICTIONS, dtype={"id": str})
    assert len(selected) == len(sample)
    assert selected["id"].reset_index(drop=True).equals(expected_sample_ids)
    assert "label" not in selected.columns
    probabilities = check_probabilities(selected, "selected ensemble")
    labels = pd.to_numeric(selected["predicted_label"], errors="raise").to_numpy(
        dtype=int
    )
    assert np.array_equal(labels, probabilities.argmax(axis=1))
    assert np.array_equal(
        selected["predicted_class"], np.asarray(DISPLAY_NAMES)[labels]
    )

    index = pd.read_csv(TASK_DIR / "submit" / "index.csv", dtype=str)
    assert index.columns.tolist() == ["id", "task4"]
    assert index["id"].reset_index(drop=True).equals(expected_sample_ids)
    assert index["task4"].notna().all()
    assert set(index["task4"].unique()).issubset(SUBMISSION_LABELS)
    expected_submission_labels = np.asarray(SUBMISSION_LABELS, dtype=object)[labels]
    assert np.array_equal(index["task4"].to_numpy(), expected_submission_labels), (
        "index.csv labels differ from the selected ensemble predictions"
    )

    ranking = pd.read_csv(TASK_DIR / "assets" / "task4-oof-ensemble-ranking.csv")
    assert len(ranking) == 28, "Expected the 28 predeclared ensemble candidates"
    assert ranking["rank"].tolist() == list(range(1, 29))
    assert ranking.loc[0, "method"] == ("soft-mean-dinov3-vitl-eva02-base-eva02-large")
    print(
        f"Task 4 artifacts verified: {len(train):,} train rows, {len(test):,} test rows, "
        f"4 aligned five-fold model OOF/test pairs, {len(ranking)} ensemble candidates, "
        "and a row-index-aligned submission. Test labels were not read."
    )


if __name__ == "__main__":
    main()
