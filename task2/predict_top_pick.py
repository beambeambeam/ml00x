"""Fit the selected Task 2 candidate on all labeled rows and predict test segments.

This creates a separate, player_id-keyed prediction artifact. It intentionally does
not edit submit/index.csv because player_id-to-ANS mapping has not been verified.
Run from task2 with: uv run python predict_top_pick.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/segment-top-pick-vip-region.joblib"
PREDICTION_PATH = ROOT / "predictions/segment-top-pick-vip-region-test.csv"
METADATA_PATH = ROOT / "assets/segment-top-pick-vip-region-test.json"


REFERENCE_DROP_COLUMNS = [
    "segment",
    "player_id",
    "random_metric_1",
    "random_metric_2",
    "random_metric_3",
]


def make_reference_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce feature engineering used by the saved reference LightGBM."""
    x = frame.drop(columns=REFERENCE_DROP_COLUMNS, errors="ignore").copy()
    ratios = {
        "spend_per_hour": ("total_spending_thb", "total_playtime_hours"),
        "spend_per_account_day": ("total_spending_thb", "account_age_days"),
        "playtime_per_account_day": ("total_playtime_hours", "account_age_days"),
        "monthly_spend_per_purchase": ("avg_monthly_spending", "spending_frequency"),
        "gifts_per_friend": ("gifts_sent_received", "friend_count"),
        "invites_per_friend": ("friend_invites_sent", "friend_count"),
        "ranked_per_team": ("ranked_participation_rate", "team_play_percentage"),
    }
    for name, (numerator, denominator) in ratios.items():
        x[name] = x[numerator] / (x[denominator].abs() + 1)
    x["weekly_minutes"] = x["play_frequency"] * x["avg_session_duration"]
    x["social_activity"] = x["friend_count"] * x["team_play_percentage"]
    x["missing_count"] = x.isna().sum(axis=1)
    return x.replace([np.inf, -np.inf], np.nan)


def make_vip_region_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the winning OOF VIP × region LightGBM features."""
    x = frame.drop(
        columns=[
            "segment",
            "player_id",
            "random_metric_1",
            "random_metric_2",
            "random_metric_3",
        ],
        errors="ignore",
    ).copy()
    ratios = {
        "spend_per_hour": ("total_spending_thb", "total_playtime_hours"),
        "spend_per_account_day": ("total_spending_thb", "account_age_days"),
        "playtime_per_account_day": ("total_playtime_hours", "account_age_days"),
        "monthly_spend_per_purchase": ("avg_monthly_spending", "spending_frequency"),
        "gifts_per_friend": ("gifts_sent_received", "friend_count"),
        "invites_per_friend": ("friend_invites_sent", "friend_count"),
        "ranked_per_team": ("ranked_participation_rate", "team_play_percentage"),
    }
    for name, (numerator, denominator) in ratios.items():
        x[name] = x[numerator] / (x[denominator].abs() + 1)
    x["weekly_minutes"] = x["play_frequency"] * x["avg_session_duration"]
    x["social_activity"] = x["friend_count"] * x["team_play_percentage"]
    vip = x["vip_tier"].map(lambda value: "__MISSING__" if pd.isna(value) else str(int(value)))
    region = x["region"].fillna("__MISSING__").astype(str)
    x["vip_region"] = vip + "__" + region
    x["missing_count"] = x.isna().sum(axis=1)
    return x.replace([np.inf, -np.inf], np.nan)


def ordered_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_, dtype=int)
    if set(classes.tolist()) != {0, 1, 2, 3}:
        raise ValueError(f"Expected classes 0-3, got {classes.tolist()}")
    return probabilities[:, np.argsort(classes)]


def validate_probabilities(name: str, probabilities: np.ndarray, rows: int) -> None:
    if probabilities.shape != (rows, 4):
        raise ValueError(f"{name} has unexpected shape {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{name} contains invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name} probability rows do not sum to one")


def main() -> None:
    train = pd.read_csv(ROOT / "data/train.csv")
    if "segment" not in train or train["segment"].isna().any():
        raise ValueError("train.csv must contain nonmissing segment labels")
    y = train["segment"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1, 2, 3}:
        raise ValueError("Expected training classes 0, 1, 2, and 3")

    # Fit only the preselected VIP × region recipe using the full labeled train set.
    candidate_x = make_vip_region_features(train)
    categorical_columns = candidate_x.select_dtypes(include=["object", "string"]).columns.tolist()
    categorical_dtypes = {}
    for column in categorical_columns:
        dtype = pd.CategoricalDtype(
            categories=sorted(candidate_x[column].dropna().unique().tolist())
        )
        candidate_x[column] = candidate_x[column].astype(dtype)
        categorical_dtypes[column] = dtype

    candidate_model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        n_estimators=440,
        learning_rate=0.063863707925569,
        num_leaves=59,
        max_depth=4,
        min_child_samples=81,
        reg_lambda=0.12942956115511214,
        colsample_bytree=0.7975990992289793,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )
    candidate_model.fit(candidate_x, y, categorical_feature=categorical_columns)

    # Reuse the already-refitted equal CatBoost/LightGBM reference ensemble.
    lightgbm_bundle = joblib.load(ROOT / "models/segment-lightgbm.joblib")
    reference_lgbm = lightgbm_bundle["model"]
    reference_catboost = CatBoostClassifier()
    reference_catboost.load_model(str(ROOT / "models/segment-catboost.cbm"))

    test = pd.read_csv(ROOT / "data/test.csv")
    if "segment" in test:
        raise ValueError("test.csv unexpectedly contains labels")
    if test["player_id"].isna().any() or not test["player_id"].is_unique:
        raise ValueError("test player_id values must be present and unique")

    reference_x = make_reference_features(test)
    reference_dtypes = lightgbm_bundle["categorical_dtypes"]
    for column, dtype in reference_dtypes.items():
        reference_x[column] = reference_x[column].astype(dtype)
    reference_lgbm_probabilities = ordered_probabilities(reference_lgbm, reference_x)

    raw_columns = list(reference_catboost.feature_names_)
    raw_test = test[raw_columns].copy()
    raw_categorical_columns = raw_test.select_dtypes(include=["object", "string"]).columns.tolist()
    raw_test[raw_categorical_columns] = (
        raw_test[raw_categorical_columns].fillna("__MISSING__").astype(str)
    )
    reference_catboost_probabilities = ordered_probabilities(reference_catboost, raw_test)
    reference_probabilities = (
        reference_lgbm_probabilities + reference_catboost_probabilities
    ) / 2.0

    candidate_test_x = make_vip_region_features(test)
    for column, dtype in categorical_dtypes.items():
        candidate_test_x[column] = candidate_test_x[column].astype(dtype)
    candidate_probabilities = ordered_probabilities(candidate_model, candidate_test_x)

    # Fixed mixture from the top-ranked OOF screen: 80% reference + 20% VIP × region.
    final_probabilities = 0.8 * reference_probabilities + 0.2 * candidate_probabilities
    validate_probabilities("reference", reference_probabilities, len(test))
    validate_probabilities("VIP-region candidate", candidate_probabilities, len(test))
    validate_probabilities("final blend", final_probabilities, len(test))
    predictions = final_probabilities.argmax(axis=1).astype(int)
    if not set(np.unique(predictions)).issubset({0, 1, 2, 3}):
        raise ValueError("Predictions must be segment labels 0-3")

    output = pd.DataFrame(
        {
            "player_id": test["player_id"].astype(str),
            **{f"prob_{label}": final_probabilities[:, label] for label in range(4)},
            "segment": predictions,
        }
    )
    if not output["player_id"].equals(test["player_id"].astype(str)):
        raise ValueError("Output player_id row alignment changed")

    # Confirm the loaded full-train reference models reproduce their saved predictions.
    old_reference = pd.read_csv(ROOT / "predictions/segment-best-test.csv")
    if not old_reference["player_id"].astype(str).equals(test["player_id"].astype(str)):
        raise ValueError("Saved reference predictions do not align with test.csv")
    reference_matches = int(
        (old_reference["segment"].to_numpy(dtype=int) == reference_probabilities.argmax(1)).sum()
    )
    if reference_matches != len(test):
        raise ValueError(
            "Saved baseline artifacts do not reproduce segment-best-test.csv: "
            f"{reference_matches:,}/{len(test):,} labels matched"
        )

    # Preserve an explicit link to the full-train model used for this prediction file.
    joblib.dump(
        {
            "model": candidate_model,
            "categorical_dtypes": categorical_dtypes,
            "categorical_columns": categorical_columns,
            "blend_weight": 0.2,
            "reference_weight": 0.8,
            "feature_recipe": "balanced_weight_vip_region_lightgbm",
        },
        MODEL_PATH,
    )
    output.to_csv(PREDICTION_PATH, index=False)
    metadata = {
        "candidate": "reference_plus_four-seed-vip-region-screen",
        "oof_macro_f1": 0.752172,
        "oof_weighted_f1": 0.765704,
        "oof_reference": "notebook/tried/34-vip-region-four-seed-average.ipynb",
        "refit_recipe": "balanced-weight VIP × region LightGBM, fitted once on all 104,000 labeled rows",
        "refit_parameters": {
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
        },
        "blend": "0.8 * saved equal CatBoost/LightGBM reference + 0.2 * full-train VIP × region model",
        "test_rows": int(len(test)),
        "prediction_file": str(PREDICTION_PATH.relative_to(ROOT)),
        "model_file": str(MODEL_PATH.relative_to(ROOT)),
        "submission_status": "not touched by this prediction script; predictions remain keyed by player_id",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(output):,} predictions to {PREDICTION_PATH.relative_to(ROOT)}")
    print(
        f"Reference argmax matches existing test predictions: {reference_matches:,}/{len(output):,}"
    )
    print(f"Predicted class counts: {output.segment.value_counts().sort_index().to_dict()}")
    print("Official submit/index.csv left unchanged; player_id-to-ANS mapping is not present.")


if __name__ == "__main__":
    main()
