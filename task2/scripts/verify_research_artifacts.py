from hashlib import sha256
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_DIR / "notebook" / "tried"
PREDICTION_DIR = PROJECT_DIR / "predictions"
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"
TEST_PATH = PROJECT_DIR / "data" / "test.csv"
REVISION = "2026-11-12"
EXPECTED_HASHES = {
    "train.csv": "0cdaa5fad6c17a7f27d965b988cd72f6113a1a53c1f88bcf5d12c25128249f48",
    "test.csv": "224345827c690d4383fa1c81d4a7baecfb22735ed80189b1d90504dfcfa9b703",
}


def verify_notebooks() -> int:
    notebooks = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]-*.ipynb"))
    if not notebooks:
        raise AssertionError(f"No research notebooks found in {NOTEBOOK_DIR}")

    executed_count = 0
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        executed = any(cell.execution_count is not None for cell in code_cells)
        if not executed and path.name != "73-realmlp-td-revised-dataset-oof-test.ipynb":
            raise AssertionError(f"No executed code cells in {path}")
        errors = [
            (cell_index, output.ename, output.evalue)
            for cell_index, cell in enumerate(code_cells)
            for output in cell.outputs
            if output.output_type == "error"
        ]
        if errors:
            raise AssertionError(f"Notebook errors in {path}: {errors}")
        executed_count += int(executed)
    return executed_count


def verify_revised_baseline() -> tuple[Path, Path, int]:
    for name, path in (("train.csv", TRAIN_PATH), ("test.csv", TEST_PATH)):
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != EXPECTED_HASHES[name]:
            raise AssertionError(f"{name} hash mismatch: {actual}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    oof_paths = sorted(
        PREDICTION_DIR.glob(f"segment-tuned-catboost-lightgbm-revised-{REVISION}-*-oof.csv")
    )
    test_paths = sorted(
        PREDICTION_DIR.glob(
            f"segment-tuned-catboost-lightgbm-revised-{REVISION}-*-test-predictions.csv"
        )
    )
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise AssertionError(
            f"Expected one revised OOF/Test file pair, got {oof_paths} / {test_paths}"
        )

    oof = pd.read_csv(oof_paths[0])
    test_predictions = pd.read_csv(test_paths[0])
    expected_train_ids = train["player_id"].astype(str)
    expected_test_ids = test["player_id"].astype(str)
    if not oof["player_id"].astype(str).equals(expected_train_ids):
        raise AssertionError("Revised OOF player_id values/order do not match revised train.csv")
    if not np.array_equal(oof["segment"].to_numpy(), train["segment"].to_numpy()):
        raise AssertionError("Revised OOF labels do not match revised train.csv")
    if not test_predictions["player_id"].astype(str).equals(expected_test_ids):
        raise AssertionError("Revised test predictions do not match revised test.csv order")
    if not set(expected_train_ids).isdisjoint(expected_test_ids):
        raise AssertionError("Train and test player_id values overlap")

    probability_columns = [
        f"{model}_prob_{class_id}"
        for model in ("catboost", "lightgbm", "blend")
        for class_id in range(4)
    ]
    for frame in (oof, test_predictions):
        missing = set(probability_columns) - set(frame.columns)
        if missing:
            raise AssertionError(f"Missing probability columns: {sorted(missing)}")
        for model in ("catboost", "lightgbm", "blend"):
            columns = [f"{model}_prob_{class_id}" for class_id in range(4)]
            probabilities = frame[columns].to_numpy(dtype=float)
            if not np.isfinite(probabilities).all() or (probabilities < 0).any():
                raise AssertionError(f"Invalid {model} probabilities")
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
                raise AssertionError(f"{model} probabilities are not row-normalized")
        if not np.array_equal(
            frame["predicted_segment"].to_numpy(),
            frame[[f"blend_prob_{class_id}" for class_id in range(4)]].to_numpy().argmax(axis=1),
        ):
            raise AssertionError("Predicted segment does not match blended probability argmax")

    if set(oof["fold"].unique()) != {1, 2, 3}:
        raise AssertionError("OOF fold assignments must contain folds 1, 2, and 3")
    model_prefix = oof_paths[0].name.removesuffix("-oof.csv")
    for suffix in ("-catboost.cbm", "-lightgbm.joblib"):
        if not (PROJECT_DIR / "models" / f"{model_prefix}{suffix}").is_file():
            raise AssertionError(f"Missing full-fit model checkpoint: {model_prefix}{suffix}")
    return oof_paths[0], test_paths[0], len(train)


def main() -> None:
    notebook_count = verify_notebooks()
    oof_path, test_path, train_rows = verify_revised_baseline()
    print(
        f"Verified {notebook_count} executed notebooks (no error outputs; the unattached, unexecuted "
        "notebook 73 is explicitly exempt); revised baseline "
        f"OOF ({train_rows:,} train rows) and Test prediction files match the {REVISION} IDs, "
        "hashes, normalized four-class probabilities, and blend argmax; model checkpoints exist. "
        f"OOF={oof_path.name}; Test={test_path.name}. Historical OOF files are excluded because "
        "they were produced from the superseded dataset snapshot."
    )


if __name__ == "__main__":
    main()
