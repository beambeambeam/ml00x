from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_DIR / "notebook" / "tried"
PREDICTION_DIR = PROJECT_DIR / "predictions"
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"


def verify_notebooks() -> int:
    notebooks = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]-*.ipynb"))
    if not notebooks:
        raise AssertionError(f"No research notebooks found in {NOTEBOOK_DIR}")

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not any(cell.execution_count is not None for cell in code_cells):
            raise AssertionError(f"No executed code cells in {path}")
        errors = [
            (cell_index, output.ename, output.evalue)
            for cell_index, cell in enumerate(code_cells)
            for output in cell.outputs
            if output.output_type == "error"
        ]
        if errors:
            raise AssertionError(f"Notebook errors in {path}: {errors}")
    return len(notebooks)


def verify_oof_predictions() -> tuple[int, int]:
    train = pd.read_csv(TRAIN_PATH)
    labels = train["segment"].to_numpy()
    probability_columns = [f"prob_{class_id}" for class_id in range(4)]
    files = sorted(PREDICTION_DIR.glob("*-oof.csv"))
    if not files:
        raise AssertionError(f"No OOF prediction CSVs found in {PREDICTION_DIR}")

    for path in files:
        predictions = pd.read_csv(path)
        missing = set([*probability_columns, "segment"]) - set(predictions.columns)
        if missing:
            raise AssertionError(f"{path} is missing required columns: {sorted(missing)}")
        if len(predictions) != len(labels):
            raise AssertionError(f"{path} has {len(predictions)} rows; expected {len(labels)}")
        if not np.array_equal(predictions["segment"].to_numpy(), labels):
            raise AssertionError(f"OOF labels are not row-aligned with train.csv: {path}")
        probabilities = predictions[probability_columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise AssertionError(f"Invalid probability values in {path}")
        if (probabilities.sum(axis=1) <= 0).any():
            raise AssertionError(f"Nonpositive probability row sum in {path}")
    return len(files), len(labels)


def main() -> None:
    notebook_count = verify_notebooks()
    prediction_count, row_count = verify_oof_predictions()
    print(
        f"Verified {notebook_count} executed research notebooks (no error outputs); "
        f"{prediction_count} OOF files row-aligned to {row_count:,} train labels "
        "with finite nonnegative four-class scores and positive row sums."
    )


if __name__ == "__main__":
    main()
