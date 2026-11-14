"""Fold-averaged Test predictions for the Task 2 top pick.

Each component is refit on the training rows of each CV fold. That fold model predicts its
held-out rows (OOF) and Test, and Test probabilities are averaged over the folds. The OOF
blend is rebuilt from the same fold models, so it should reproduce the saved OOF scores.

Recipe (same as the top pick in predict_top_pick.py, but fold-averaged):
    0.8 * reference + 0.2 * mean over seeds {42, 2025, 1337, 31415} of VIP x region LightGBM
    reference = equal average of tuned CatBoost and tuned LightGBM (split seed 42)

Writes predictions/segment-top-pick-foldavg-test.csv keyed by player_id. Does not touch
submit/index.csv. Run from task2 with: uv run python predict_top_pick_foldavg.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from predict_top_pick import make_reference_features, make_vip_region_features

ROOT = Path(__file__).resolve().parent
PREDICTION_PATH = ROOT / "predictions/segment-top-pick-foldavg-test.csv"
METADATA_PATH = ROOT / "assets/segment-top-pick-foldavg-test.json"

N_SPLITS = 3
REFERENCE_SPLIT_SEED = 42
VIP_SPLIT_SEEDS = {
    42: "predictions/segment-vip-region-balanced-oof.csv",
    2025: "predictions/segment-vip-region-seed-2025-oof.csv",
    1337: "predictions/segment-vip-region-seed-1337-oof.csv",
    31415: "predictions/segment-vip-region-seed-31415-oof.csv",
}
REFERENCE_OOF = "predictions/segment-oof.csv"
TOP_PICK_TEST = "predictions/segment-top-pick-vip-region-test.csv"

# Tuned LightGBM from player-segment-tuned.ipynb (best Optuna trial, 440 trees).
REFERENCE_LGB = {
    "learning_rate": 0.063863707925569,
    "num_leaves": 59,
    "max_depth": 4,
    "min_child_samples": 81,
    "reg_lambda": 0.12942956115511214,
    "colsample_bytree": 0.7975990992289793,
}
# Balanced-weight VIP x region LightGBM (notebooks 28-33).
VIP_LGB = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 440,
    "learning_rate": 0.063863707925569,
    "num_leaves": 59,
    "max_depth": 4,
    "min_child_samples": 81,
    "reg_lambda": 0.12942956115511214,
    "colsample_bytree": 0.7975990992289793,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
}


def encode_pair(fit: pd.DataFrame, other: pd.DataFrame, categorical: list[str]):
    """Fix category vocabularies on the fit rows; unseen categories become missing."""
    a, b = fit.copy(), other.copy()
    for col in categorical:
        dtype = pd.CategoricalDtype(categories=sorted(a[col].dropna().unique().tolist()))
        a[col] = a[col].astype(dtype)
        b[col] = b[col].astype(dtype)
    return a, b


def check_probabilities(name: str, probs: np.ndarray, rows: int) -> None:
    assert probs.shape == (rows, 4), (name, probs.shape)
    assert np.isfinite(probs).all() and (probs >= 0).all(), name
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), name


def main() -> None:
    train = pd.read_csv(ROOT / "data/train.csv")
    y = train["segment"].to_numpy(dtype=int)
    test = pd.read_csv(ROOT / "data/test.csv")
    assert "segment" not in test and test["player_id"].is_unique
    n_train, n_test = len(train), len(test)

    # Reference: tuned LightGBM + CatBoost, split seed 42 (same folds as segment-oof.csv).
    x_ref = make_reference_features(train)
    t_ref = make_reference_features(test)
    ref_cat = x_ref.select_dtypes(include=["object", "string"]).columns.tolist()

    raw = train.drop(columns=["segment", "player_id"]).copy()
    raw_cat = raw.select_dtypes(include=["object", "string"]).columns.tolist()
    raw[raw_cat] = raw[raw_cat].fillna("__MISSING__").astype(str)
    t_raw = test[raw.columns].copy()
    t_raw[raw_cat] = t_raw[raw_cat].fillna("__MISSING__").astype(str)

    lgb_oof = np.zeros((n_train, 4))
    cat_oof = np.zeros((n_train, 4))
    lgb_test = np.zeros((n_test, 4))
    cat_test = np.zeros((n_test, 4))
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=REFERENCE_SPLIT_SEED)
    for k, (tr, va) in enumerate(folds.split(x_ref, y), start=1):
        a, b_va = encode_pair(x_ref.iloc[tr], x_ref.iloc[va], ref_cat)
        _, b_te = encode_pair(x_ref.iloc[tr], t_ref, ref_cat)
        lgb = LGBMClassifier(
            n_estimators=440,
            objective="multiclass",
            random_state=42,
            n_jobs=4,
            verbosity=-1,
            **REFERENCE_LGB,
        )
        lgb.fit(a, y[tr])
        lgb_oof[va] = lgb.predict_proba(b_va)
        lgb_test += lgb.predict_proba(b_te) / N_SPLITS

        cat = CatBoostClassifier(
            iterations=300,
            depth=7,
            learning_rate=0.08,
            loss_function="MultiClass",
            random_seed=42,
            thread_count=4,
            allow_writing_files=False,
            verbose=False,
        )
        cat.fit(raw.iloc[tr], y[tr], cat_features=raw_cat)
        cat_oof[va] = cat.predict_proba(raw.iloc[va])
        cat_test += cat.predict_proba(t_raw) / N_SPLITS
        print(f"reference fold {k} done", flush=True)

    ref_oof = (lgb_oof + cat_oof) / 2
    ref_test = (lgb_test + cat_test) / 2
    check_probabilities("reference oof", ref_oof, n_train)
    check_probabilities("reference test", ref_test, n_test)

    # VIP x region LightGBM: one fold-averaged model per split seed.
    x_vip = make_vip_region_features(train)
    t_vip = make_vip_region_features(test)
    vip_cat = x_vip.select_dtypes(include=["object", "string"]).columns.tolist()
    vip_oof_by_seed: dict[int, np.ndarray] = {}
    vip_test_by_seed: dict[int, np.ndarray] = {}
    for seed in VIP_SPLIT_SEEDS:
        oof = np.zeros((n_train, 4))
        test_probs = np.zeros((n_test, 4))
        folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        for tr, va in folds.split(x_vip, y):
            a, b_va = encode_pair(x_vip.iloc[tr], x_vip.iloc[va], vip_cat)
            _, b_te = encode_pair(x_vip.iloc[tr], t_vip, vip_cat)
            model = LGBMClassifier(**VIP_LGB)
            model.fit(a, y[tr], categorical_feature=vip_cat)
            oof[va] = model.predict_proba(b_va)
            test_probs += model.predict_proba(b_te) / N_SPLITS
        vip_oof_by_seed[seed] = oof
        vip_test_by_seed[seed] = test_probs
        print(f"VIP x region split seed {seed} done", flush=True)

    vip_oof = np.mean(list(vip_oof_by_seed.values()), axis=0)
    vip_test = np.mean(list(vip_test_by_seed.values()), axis=0)
    final_oof = 0.8 * ref_oof + 0.2 * vip_oof
    final_test = 0.8 * ref_test + 0.2 * vip_test
    check_probabilities("final test", final_test, n_test)

    # Reproduction checks against the saved OOF files.
    def load(path: str) -> np.ndarray:
        frame = pd.read_csv(ROOT / path)
        assert np.array_equal(frame["segment"].to_numpy(), y), path
        return frame[[f"prob_{i}" for i in range(4)]].to_numpy(dtype=float)

    saved_reference = load(REFERENCE_OOF)
    reproduction = {
        "reference_max_abs_diff": float(np.abs(ref_oof - saved_reference).max()),
        "vip_seed_max_abs_diff": {
            str(seed): float(np.abs(vip_oof_by_seed[seed] - load(path)).max())
            for seed, path in VIP_SPLIT_SEEDS.items()
        },
    }
    oof_scores = {
        "reference_macro_f1": f1_score(y, ref_oof.argmax(1), average="macro"),
        "top_pick_macro_f1": f1_score(y, final_oof.argmax(1), average="macro"),
        "top_pick_weighted_f1": f1_score(y, final_oof.argmax(1), average="weighted"),
    }
    print("OOF:", json.dumps(oof_scores, indent=2))
    print("Reproduction:", json.dumps(reproduction, indent=2))

    # Compare with the full-refit top pick already saved.
    full_refit = pd.read_csv(ROOT / TOP_PICK_TEST)
    full_refit_labels = full_refit["segment"].to_numpy(dtype=int)
    fold_labels = final_test.argmax(axis=1).astype(int)

    output = pd.DataFrame(
        {
            "player_id": test["player_id"].astype(str),
            **{f"prob_{label}": final_test[:, label] for label in range(4)},
            "segment": fold_labels,
        }
    )
    assert output["player_id"].equals(test["player_id"].astype(str))
    assert len(output) == n_test and set(np.unique(fold_labels)) <= {0, 1, 2, 3}
    output.to_csv(PREDICTION_PATH, index=False)

    metadata = {
        "candidate": "top pick, fold-averaged Test predictions",
        "recipe": "0.8 * (CatBoost + LightGBM)/2 + 0.2 * mean over VIP x region seeds 42, 2025, 1337, 31415",
        "cv": "StratifiedKFold(3, shuffle=True); reference split seed 42; VIP splits as listed",
        "test_rows": int(n_test),
        "oof_scores": oof_scores,
        "reproduction": reproduction,
        "agreement_with_full_refit_top_pick": float((fold_labels == full_refit_labels).mean()),
        "differing_labels_vs_full_refit": int((fold_labels != full_refit_labels).sum()),
        "predicted_class_counts": {
            str(k): int(v) for k, v in pd.Series(fold_labels).value_counts().sort_index().items()
        },
        "prediction_file": str(PREDICTION_PATH.relative_to(ROOT)),
        "submission_status": "not written to submit/index.csv",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(output):,} rows to {PREDICTION_PATH.relative_to(ROOT)}")
    print(f"Agreement with full-refit top pick: {metadata['agreement_with_full_refit_top_pick']:.4%}")


if __name__ == "__main__":
    main()
