"""Task 4: game-title image classification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent
DATA, PREDICTIONS, SUBMIT = ROOT / "data", ROOT / "predictions", ROOT / "submit"
RANDOM_STATE = 42
LABELS = ["7-11k", "fiveN", "freefried", "pube", "roblock"]


def image_features(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((64, 32), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    pooled = rgb.reshape(8, 4, 8, 8, 3).mean(axis=(1, 3)).ravel()
    small = np.asarray(image.resize((16, 8), Image.Resampling.BILINEAR), dtype=np.float32).ravel() / 255.0
    summary = np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)), np.quantile(rgb, [0.1, 0.5, 0.9], axis=(0, 1)).ravel()])
    # Fast edge summaries capture UI/text density without a heavy dependency.
    edges = np.concatenate([np.abs(np.diff(gray, axis=0)).mean(axis=0), np.abs(np.diff(gray, axis=1)).mean(axis=1)])
    return np.concatenate([pooled, small, summary, edges]).astype(np.float32)


def make_features(ids: pd.Series, folder: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)["X"]
    paths = [folder / f"{image_id}.jpg" for image_id in ids]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = []
        for n, row in enumerate(pool.map(image_features, paths), 1):
            rows.append(row)
            if n % 2000 == 0:
                print(f"extracted {n:,}/{len(paths):,} from {folder.name}", flush=True)
    X = np.vstack(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X)
    return X


def build_model():
    return make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=2e-5, class_weight="balanced", max_iter=40, tol=1e-3, random_state=RANDOM_STATE, n_jobs=-1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    PREDICTIONS.mkdir(exist_ok=True)
    SUBMIT.mkdir(exist_ok=True)
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    train["class"] = train["class"].str.strip()
    y = train["class"].to_numpy()
    X = make_features(train["id"], DATA / "train", PREDICTIONS / "train-features.npz")
    Xt = make_features(test["id"], DATA / "test", PREDICTIONS / "test-features.npz")
    print(f"features: train={X.shape}, test={Xt.shape}")
    if not args.skip_validation:
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        check = build_model().fit(X[tr], y[tr])
        pred = check.predict(X[va])
        print(f"validation macro-F1: {f1_score(y[va], pred, average='macro'):.5f}")
        print(classification_report(y[va], pred, labels=LABELS, zero_division=0))
    model = build_model().fit(X, y)
    pred = model.predict(Xt)
    pred = np.where(pred == "7-11k", "7-11K", pred)
    pd.DataFrame({"id": test["id"], "task4": pred}).to_csv(SUBMIT / "index.csv", index=False)
    pd.DataFrame({"id": test["id"], "prediction": pred}).to_csv(PREDICTIONS / "test-predictions.csv", index=False)
    print(f"wrote {len(pred):,} predictions to {SUBMIT / 'index.csv'}")


if __name__ == "__main__":
    main()
