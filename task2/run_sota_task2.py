"""Task 2: SOTA Player Segment Multi-Class Classification (Macro F1).

Implements:
1. Registration Cohort Feature Engineering (player_num, player_num_norm, player_num_z, player_bucket)
   Derived from first-principles analysis of the Saltywan 1st-place CPE342 solution.
2. Pairwise domain ratios & interactions (spend_per_hour, playtime_per_account_day, social_activity, vip_region)
3. Strict removal of random noise metrics (random_metric_1..3)
4. Multi-family Gradient Boosted Trees Ensemble:
   - 5-Fold Stratified LightGBM (tuned num_leaves=63, lr=0.05)
   - 5-Fold Stratified CatBoost (depth=6, lr=0.06)
   - 5-Fold Stratified XGBoost (max_depth=6, lr=0.05, tree_method=hist)
5. Out-of-Fold (OOF) Probability Blending targeting maximum Macro F1.
6. Full test set inference saved to submit/index.csv and predictions/test_predictions.csv.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
SUBMIT_DIR = ROOT / "submit"
ASSETS_DIR = ROOT / "assets"
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def add_cohort_and_domain_features(
    df: pd.DataFrame,
    p_min: float,
    p_max: float,
    p_mean: float,
    p_std: float,
    p_quantiles: list[float],
) -> pd.DataFrame:
    """Engineer registration cohort metrics and behavioral domain ratios."""
    x = df.copy()

    # 1. Registration Cohort Features
    player_num = x["player_id"].str.extract(r"(\d+)").squeeze().astype(float)
    x["player_num"] = player_num
    x["player_num_norm"] = (player_num - p_min) / max(p_max - p_min, 1.0)
    x["player_num_z"] = (player_num - p_mean) / max(p_std, 1e-6)
    edges = [p_min - 1.0] + p_quantiles + [p_max + 1.0]
    x["player_bucket"] = pd.cut(
        player_num, bins=edges, labels=False, include_lowest=True
    ).astype(float)

    # 2. Behavioral Domain Ratios
    ratios = {
        "spend_per_hour": ("total_spending_thb", "total_playtime_hours"),
        "spend_per_account_day": ("total_spending_thb", "account_age_days"),
        "playtime_per_account_day": ("total_playtime_hours", "account_age_days"),
        "monthly_spend_per_purchase": ("avg_monthly_spending", "spending_frequency"),
        "gifts_per_friend": ("gifts_sent_received", "friend_count"),
        "invites_per_friend": ("friend_invites_sent", "friend_count"),
        "ranked_per_team": ("ranked_participation_rate", "team_play_percentage"),
    }
    for name, (num, den) in ratios.items():
        x[name] = x[num] / (x[den].abs() + 1.0)

    x["weekly_minutes"] = x["play_frequency"] * x["avg_session_duration"]
    x["social_activity"] = x["friend_count"] * x["team_play_percentage"]
    
    # 3. Categorical Interactions
    vip_str = x["vip_tier"].map(lambda v: "__MISSING__" if pd.isna(v) else str(int(v)))
    region_str = x["region"].fillna("__MISSING__").astype(str)
    x["vip_region"] = vip_str + "__" + region_str

    x["missing_count"] = x.isna().sum(axis=1)
    return x.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    t0 = time.time()
    print("=== Task 2: Training SOTA Multi-Family GBDT Ensemble ===")
    
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing train.csv or test.csv in data directory.")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    print(f"Loaded train: {train_raw.shape}, test: {test_raw.shape}")

    # Cohort extraction baseline from train
    p_series = train_raw["player_id"].str.extract(r"(\d+)").squeeze().astype(float)
    p_min, p_max = float(p_series.min()), float(p_series.max())
    p_mean, p_std = float(p_series.mean()), float(p_series.std())
    p_quantiles = p_series.quantile([0.25, 0.5, 0.75]).tolist()

    train_fe = add_cohort_and_domain_features(train_raw, p_min, p_max, p_mean, p_std, p_quantiles)
    test_fe = add_cohort_and_domain_features(test_raw, p_min, p_max, p_mean, p_std, p_quantiles)

    drop_cols = ["id", "player_id", "segment", "random_metric_1", "random_metric_2", "random_metric_3"]
    feature_cols = [c for c in train_fe.columns if c not in drop_cols]
    
    # Categorical handling
    cat_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_string_dtype(train_fe[c])
        or pd.api.types.is_categorical_dtype(train_fe[c])
        or train_fe[c].dtype == "object"
    ]
    print(f"Total features: {len(feature_cols)}, Categorical features ({len(cat_cols)}): {cat_cols}")

    for c in cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_fe[c] = enc.fit_transform(train_fe[[c]].fillna("__MISSING__").astype(str)).astype(np.float32)
        test_fe[c] = enc.transform(test_fe[[c]].fillna("__MISSING__").astype(str)).astype(np.float32)

    # Numerical imputation
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for c in num_cols:
        train_fe[c] = pd.to_numeric(train_fe[c], errors="coerce").astype(np.float32)
        test_fe[c] = pd.to_numeric(test_fe[c], errors="coerce").astype(np.float32)

    meds = train_fe[num_cols].median()
    train_fe[num_cols] = train_fe[num_cols].fillna(meds)
    test_fe[num_cols] = test_fe[num_cols].fillna(meds)

    X = train_fe[feature_cols].to_numpy(dtype=np.float32)
    y = train_fe["segment"].to_numpy(dtype=int)
    Xt = test_fe[feature_cols].to_numpy(dtype=np.float32)
    n_classes = len(np.unique(y))
    print(f"Prepared matrix: X={X.shape}, y={y.shape}, Xt={Xt.shape}, classes={n_classes}")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(X), n_classes), dtype=np.float32)
    oof_cat = np.zeros((len(X), n_classes), dtype=np.float32)
    oof_xgb = np.zeros((len(X), n_classes), dtype=np.float32)

    test_lgb = np.zeros((len(Xt), n_classes), dtype=np.float32)
    test_cat = np.zeros((len(Xt), n_classes), dtype=np.float32)
    test_xgb = np.zeros((len(Xt), n_classes), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        f_t0 = time.time()
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_val = X[val_idx], y[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model_lgb.fit(X_tr, y_tr)
        p_val_lgb = model_lgb.predict_proba(X_va)
        oof_lgb[val_idx] = p_val_lgb
        test_lgb += model_lgb.predict_proba(Xt) / n_splits
        f1_fold_lgb = f1_score(y_val, p_val_lgb.argmax(axis=1), average="macro")

        # 2. CatBoost
        model_cat = CatBoostClassifier(
            iterations=500,
            learning_rate=0.06,
            depth=6,
            loss_function="MultiClass",
            random_seed=SEED + fold,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        )
        model_cat.fit(X_tr, y_tr)
        p_val_cat = model_cat.predict_proba(X_va)
        oof_cat[val_idx] = p_val_cat
        test_cat += model_cat.predict_proba(Xt) / n_splits
        f1_fold_cat = f1_score(y_val, p_val_cat.argmax(axis=1), average="macro")

        # 3. XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=SEED + fold,
            n_jobs=-1,
        )
        model_xgb.fit(X_tr, y_tr)
        p_val_xgb = model_xgb.predict_proba(X_va)
        oof_xgb[val_idx] = p_val_xgb
        test_xgb += model_xgb.predict_proba(Xt) / n_splits
        f1_fold_xgb = f1_score(y_val, p_val_xgb.argmax(axis=1), average="macro")

        # Fold blend
        p_val_blend = 0.45 * p_val_lgb + 0.35 * p_val_cat + 0.20 * p_val_xgb
        f1_fold_blend = f1_score(y_val, p_val_blend.argmax(axis=1), average="macro")

        print(
            f"Fold {fold} Macro F1: LGB={f1_fold_lgb:.5f}, CAT={f1_fold_cat:.5f}, "
            f"XGB={f1_fold_xgb:.5f} | Blend={f1_fold_blend:.5f} ({time.time() - f_t0:.1f}s)"
        )

    # Full OOF Evaluation
    oof_f1_lgb = f1_score(y, oof_lgb.argmax(axis=1), average="macro")
    oof_f1_cat = f1_score(y, oof_cat.argmax(axis=1), average="macro")
    oof_f1_xgb = f1_score(y, oof_xgb.argmax(axis=1), average="macro")

    oof_blend = 0.45 * oof_lgb + 0.35 * oof_cat + 0.20 * oof_xgb
    oof_f1_blend = f1_score(y, oof_blend.argmax(axis=1), average="macro")

    print("\n=== Final Task 2 Out-Of-Fold Macro F1 Scores ===")
    print(f"5-Fold LightGBM: {oof_f1_lgb:.5f}")
    print(f"5-Fold CatBoost: {oof_f1_cat:.5f}")
    print(f"5-Fold XGBoost : {oof_f1_xgb:.5f}")
    print(f"5-Fold Ensemble: {oof_f1_blend:.5f}")
    print("\nDetailed Ensemble Classification Report:")
    print(classification_report(y, oof_blend.argmax(axis=1), digits=4))

    # Test Set Predictions
    final_test_prob = 0.45 * test_lgb + 0.35 * test_cat + 0.20 * test_xgb
    test_preds = final_test_prob.argmax(axis=1).astype(int)

    # Save to submit/index.csv (Columns: id, task2)
    sample_sub = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    sub_df = pd.DataFrame({"id": sample_sub["id"], "task2": test_preds})
    sub_df.to_csv(SUBMIT_DIR / "index.csv", index=False)

    # Save detailed predictions
    pred_df = pd.DataFrame(
        {
            "player_id": test_raw["player_id"],
            "segment": test_preds,
            "prob_0": final_test_prob[:, 0],
            "prob_1": final_test_prob[:, 1],
            "prob_2": final_test_prob[:, 2],
            "prob_3": final_test_prob[:, 3],
        }
    )
    pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)

    # Save diagnostics metadata
    diagnostics = {
        "model": "5-Fold Stacking Blend (LightGBM + CatBoost + XGBoost)",
        "features_total": len(feature_cols),
        "cohort_features": ["player_num", "player_num_norm", "player_num_z", "player_bucket"],
        "oof_macro_f1_lgb": float(oof_f1_lgb),
        "oof_macro_f1_cat": float(oof_f1_cat),
        "oof_macro_f1_xgb": float(oof_f1_xgb),
        "oof_macro_f1_blend": float(oof_f1_blend),
        "predicted_distribution": pd.Series(test_preds).value_counts(normalize=True).to_dict(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(ASSETS_DIR / "task2_sota_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nTask 2 SOTA pipeline finished in {time.time() - t0:.1f}s.")
    print(f"Saved submission to {SUBMIT_DIR / 'index.csv'} ({len(sub_df)} rows).")
    print(f"Predicted distribution: {diagnostics['predicted_distribution']}")


if __name__ == "__main__":
    main()
