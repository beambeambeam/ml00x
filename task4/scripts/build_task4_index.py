from pathlib import Path

import numpy as np
import pandas as pd

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parent
PREDICTIONS = (
    TASK_DIR
    / "predictions"
    / (
        "task4-f1-0p9911-soft-mean-dinov3-vitl-eva02-base-eva02-large-test-predictions.csv"
    )
)
SAMPLE = REPO_ROOT / "submission" / "sample_submission.csv"
OUTPUT = TASK_DIR / "submit" / "index.csv"
SUBMISSION_LABELS = ["7-11K", "fiveN", "freefried", "pube", "roblock"]
DISPLAY_NAMES = ["7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock"]


def main() -> None:
    sample_ids = pd.read_csv(SAMPLE, usecols=["id"], dtype={"id": str})["id"]
    test_rows = pd.read_csv(
        TASK_DIR / "data" / "test.csv", usecols=["id"], dtype={"id": str}
    )
    predictions = pd.read_csv(PREDICTIONS, dtype={"id": str})
    assert sample_ids.is_unique and predictions["id"].is_unique
    assert len(test_rows) == len(sample_ids) == len(predictions), (
        "Test metadata, sample submission, and predictions must have equal row counts"
    )
    expected_ids = pd.Series(
        [f"ANS{row_number:05d}" for row_number in range(1, len(sample_ids) + 1)],
        name="id",
    )
    assert sample_ids.reset_index(drop=True).equals(expected_ids), (
        "Sample IDs must encode test row order as ANS00001, ANS00002, ..."
    )
    assert predictions["id"].reset_index(drop=True).equals(expected_ids), (
        "Test predictions are not in sample-submission row-index order"
    )
    assert "label" not in predictions.columns
    assert {"id", "predicted_label"}.issubset(predictions.columns)

    label_values = pd.to_numeric(predictions["predicted_label"], errors="raise")
    label_array = label_values.to_numpy(dtype=np.float64)
    assert np.isfinite(label_array).all()
    assert np.equal(label_array, np.floor(label_array)).all()
    labels = label_array.astype(np.int64)
    assert ((0 <= labels) & (labels < len(SUBMISSION_LABELS))).all()
    if "predicted_class" in predictions:
        expected_names = np.asarray(DISPLAY_NAMES)[labels]
        assert np.array_equal(predictions["predicted_class"].to_numpy(), expected_names)

    # ANS identifiers encode each test row's position; retain the verified order.
    index = pd.DataFrame(
        {"id": sample_ids, "task4": np.asarray(SUBMISSION_LABELS, dtype=object)[labels]}
    )
    assert index["task4"].notna().all()
    assert set(index["task4"].unique()).issubset(SUBMISSION_LABELS)

    temp_path = OUTPUT.with_suffix(".csv.tmp")
    index.to_csv(temp_path, index=False)
    temp_path.replace(OUTPUT)

    saved = pd.read_csv(OUTPUT, dtype={"id": str, "task4": str})
    assert saved.columns.tolist() == ["id", "task4"]
    assert saved["id"].equals(sample_ids)
    assert len(saved) == len(sample_ids)
    assert saved["task4"].notna().all()
    print(f"Wrote {len(saved):,} validated Task 4 rows to {OUTPUT}")
    print(saved["task4"].value_counts().to_string())


if __name__ == "__main__":
    main()
