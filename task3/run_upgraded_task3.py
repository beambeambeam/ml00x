"""
Task 3: Promoted Champion Architecture with Bayes-Optimal Conditional Quantile Hurdle
Hybrid Two-Stage Dynamic Quantile Hurdle + Scale-Relative Log-Multiplier Hurdle + Direct Raw-Scale L1 Stacking + Level-2 LGBM(MAE) Meta-Stacker.
Strictly fold-local preprocessing (zero cross-fold leakage).
"""
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
SUBMIT_DIR = ROOT / "submit"
ASSET_DIR = ROOT / "assets"


def compute_nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_c = np.clip(y_pred, 0.0, None)
    denom = np.sum(y_true)
    if denom <= 0:
        return 0.0
    return float(1.0 / (1.0 + (np.sum(np.abs(y_true - y_pred_c)) / denom)))


def estimate_prev_hist_ratio(df: pd.DataFrame) -> float:
    calc_hist = df["total_transactions"] * df["avg_transaction_value"]
    hist_clean = df["historical_spending"].copy()
    mask_hist = hist_clean.isna() & calc_hist.notna()
    hist_clean[mask_hist] = calc_hist[mask_hist]
    valid = (hist_clean > 0) & df["prev_month_spending"].notna()
    if not np.any(valid):
        return 0.1636
    return float((df.loc[valid, "prev_month_spending"] / hist_clean.loc[valid]).median())


def engineer_features(
    df: pd.DataFrame, prev_to_hist_ratio: float, medians: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    d = df.copy()
    drop_cols = [c for c in ["player_id", "id", "spending_30d", "is_spender"] if c in d.columns]
    if drop_cols:
        d = d.drop(columns=drop_cols)

    d["is_na_hist"] = d["historical_spending"].isna().astype(np.float32)
    d["is_na_prev"] = d["prev_month_spending"].isna().astype(np.float32)
    d["is_na_tx"] = d["total_transactions"].isna().astype(np.float32)
    d["is_na_avg_tx"] = d["avg_transaction_value"].isna().astype(np.float32)

    calc_hist = d["total_transactions"] * d["avg_transaction_value"]
    hist_clean = d["historical_spending"].copy()
    mask_hist = hist_clean.isna() & calc_hist.notna()
    hist_clean[mask_hist] = calc_hist[mask_hist]
    d["hist_clean"] = hist_clean

    prev_clean = d["prev_month_spending"].copy()
    mask_prev = prev_clean.isna() & d["hist_clean"].notna()
    prev_clean[mask_prev] = d.loc[mask_prev, "hist_clean"] * prev_to_hist_ratio
    d["prev_clean"] = prev_clean

    d["expected_monthly_burn"] = d["hist_clean"] / (d["account_age_days"] / 30.0 + 1.0)
    d["spending_momentum"] = d["prev_clean"] / (d["expected_monthly_burn"] + 10.0)
    d["spending_delta"] = d["prev_clean"] - d["expected_monthly_burn"]
    d["tx_velocity"] = d["total_transactions"] / (d["account_age_days"] / 30.0 + 1.0)
    d["expected_30d_from_freq"] = d["purchase_frequency"] * d["avg_transaction_value"]
    d["recency_purchase_decay"] = np.exp(-d["days_since_last_purchase"].fillna(365.0) / 30.0)
    d["recency_login_decay"] = np.exp(-d["days_since_last_login"].fillna(365.0) / 14.0)
    d["rfm_score"] = d["prev_clean"] * d["recency_purchase_decay"] * d["recency_login_decay"]
    d["weekly_play_minutes"] = d["avg_session_length"] * d["sessions_per_week"]
    d["whale_vip_anchor"] = (
        d["prev_clean"]
        * (1.0 + 0.15 * d["vip_status"].fillna(0.0))
        * (1.0 + 0.1 * d["segment"].fillna(0.0))
    )
    d["season_adj_prev"] = d["prev_clean"] * (1.0 + 0.1 * d["seasonal_spending_pattern"].fillna(0.0))

    d = d.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = d.median()
    d_imputed = d.fillna(medians)
    return d, d_imputed, medians


def interpolate_quantile_predictions(
    p_cal: np.ndarray, quantiles: list[float], pred_q_dict: dict[float, np.ndarray], tau: float = 0.42
) -> np.ndarray:
    q_star = np.clip((p_cal - tau) / np.maximum(p_cal, 1e-4), quantiles[0], quantiles[-1])
    n = len(p_cal)
    out = np.zeros(n, dtype=np.float64)
    q_arr = np.array(quantiles)
    for i in range(n):
        if p_cal[i] < tau:
            out[i] = 0.0
        else:
            qs = q_star[i]
            idx = np.searchsorted(q_arr, qs)
            if idx == 0:
                out[i] = pred_q_dict[q_arr[0]][i]
            elif idx >= len(q_arr):
                out[i] = pred_q_dict[q_arr[-1]][i]
            else:
                q_lo, q_hi = q_arr[idx - 1], q_arr[idx]
                w_hi = (qs - q_lo) / (q_hi - q_lo)
                out[i] = (1.0 - w_hi) * pred_q_dict[q_lo][i] + w_hi * pred_q_dict[q_hi][i]
    return out


def main():
    t0 = time.time()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    y = train_df["spending_30d"].to_numpy(np.float64)
    y_bin = (y > 0).astype(np.int32)
    strat_key = y_bin.astype(str) + "_" + train_df["segment"].fillna(-1).astype(int).astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n = len(train_df)
    n_te = len(test_df)

    p_oof = np.zeros(n, dtype=np.float64)
    p_test = np.zeros(n_te, dtype=np.float64)

    hurdle_lgb_pos_oof = np.zeros(n, dtype=np.float64)
    hurdle_cb_pos_oof = np.zeros(n, dtype=np.float64)
    hurdle_xgb_pos_oof = np.zeros(n, dtype=np.float64)

    hurdle_lgb_pos_test = np.zeros(n_te, dtype=np.float64)
    hurdle_cb_pos_test = np.zeros(n_te, dtype=np.float64)
    hurdle_xgb_pos_test = np.zeros(n_te, dtype=np.float64)

    # Quantile bank
    quantiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    q_oof_dict = {q: np.zeros(n, dtype=np.float64) for q in quantiles}
    q_test_dict = {q: np.zeros(n_te, dtype=np.float64) for q in quantiles}

    raw_lgb_oof = np.zeros(n, dtype=np.float64)
    raw_cb_oof = np.zeros(n, dtype=np.float64)
    raw_xgb_oof = np.zeros(n, dtype=np.float64)

    raw_lgb_test = np.zeros(n_te, dtype=np.float64)
    raw_cb_test = np.zeros(n_te, dtype=np.float64)
    raw_xgb_test = np.zeros(n_te, dtype=np.float64)

    anchor_prev_oof = np.zeros(n, dtype=np.float64)
    anchor_hist_oof = np.zeros(n, dtype=np.float64)
    anchor_prev_test = np.zeros(n_te, dtype=np.float64)
    anchor_hist_test = np.zeros(n_te, dtype=np.float64)

    print("=== Training Task 3 Bayes-Optimal Quantile Hurdle + Multi-Model Stacking (5 Folds) ===")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_key), 1):
        ft0 = time.time()
        f_tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        f_va_df = train_df.iloc[va_idx].reset_index(drop=True)
        ratio = estimate_prev_hist_ratio(f_tr_df)

        X_tr_raw, X_tr_imp, med = engineer_features(f_tr_df, ratio, None)
        X_va_raw, X_va_imp, _ = engineer_features(f_va_df, ratio, med)
        X_te_raw, X_te_imp, _ = engineer_features(test_df, ratio, med)

        y_tr = y[tr_idx]
        y_bin_tr = y_bin[tr_idx]

        anchor_tr = np.maximum(X_tr_imp["prev_clean"].to_numpy(), 100.0)
        anchor_va = np.maximum(X_va_imp["prev_clean"].to_numpy(), 100.0)
        anchor_te = np.maximum(X_te_imp["prev_clean"].to_numpy(), 100.0)

        anchor_prev_oof[va_idx] = X_va_imp["prev_clean"].to_numpy()
        anchor_hist_oof[va_idx] = X_va_imp["hist_clean"].to_numpy()
        anchor_prev_test += X_te_imp["prev_clean"].to_numpy() / 5.0
        anchor_hist_test += X_te_imp["hist_clean"].to_numpy() / 5.0

        # 1. Stage 1 Classifiers (LGBM + CatBoost + XGBoost)
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.03, num_leaves=63,
            subsample_for_bin=50000, colsample_bytree=0.8, subsample=0.8,
            min_child_samples=40, random_state=42 + fold, n_jobs=-1, verbose=-1
        )
        clf_lgb.fit(X_tr_raw, y_bin_tr)
        p_lgb_va = clf_lgb.predict_proba(X_va_raw)[:, 1]
        p_lgb_te = clf_lgb.predict_proba(X_te_raw)[:, 1]

        clf_cb = CatBoostClassifier(
            iterations=900, learning_rate=0.04, depth=6,
            random_seed=42 + fold, thread_count=-1, verbose=0, allow_writing_files=False
        )
        clf_cb.fit(X_tr_imp, y_bin_tr)
        p_cb_va = clf_cb.predict_proba(X_va_imp)[:, 1]
        p_cb_te = clf_cb.predict_proba(X_te_imp)[:, 1]

        clf_xgb = xgb.XGBClassifier(
            n_estimators=700, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=25,
            tree_method="hist", eval_metric="logloss", random_state=42 + fold, n_jobs=-1
        )
        clf_xgb.fit(X_tr_raw, y_bin_tr)
        p_xgb_va = clf_xgb.predict_proba(X_va_raw)[:, 1]
        p_xgb_te = clf_xgb.predict_proba(X_te_raw)[:, 1]

        p_oof[va_idx] = 0.45 * p_lgb_va + 0.35 * p_cb_va + 0.20 * p_xgb_va
        p_test += (0.45 * p_lgb_te + 0.35 * p_cb_te + 0.20 * p_xgb_te) / 5.0

        # 2. Stage 2 Positive Regressors (Log-multiplier Hurdle)
        pos_mask = y_tr > 0
        z_tr = np.log1p(y_tr[pos_mask] / anchor_tr[pos_mask])

        reg_hurdle_lgb = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=950, learning_rate=0.03,
            num_leaves=63, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, random_state=42 + fold, n_jobs=-1, verbose=-1
        )
        reg_hurdle_lgb.fit(X_tr_raw.iloc[pos_mask], z_tr)
        hurdle_lgb_pos_oof[va_idx] = np.clip(anchor_va * np.expm1(np.clip(reg_hurdle_lgb.predict(X_va_raw), 0, 10)), 0, None)
        hurdle_lgb_pos_test += np.clip(anchor_te * np.expm1(np.clip(reg_hurdle_lgb.predict(X_te_raw), 0, 10)), 0, None) / 5.0

        reg_hurdle_cb = CatBoostRegressor(
            loss_function="MAE", iterations=1200, learning_rate=0.04, depth=6,
            l2_leaf_reg=5.0, random_seed=42 + fold, thread_count=-1, verbose=0, allow_writing_files=False
        )
        reg_hurdle_cb.fit(X_tr_imp.iloc[pos_mask], z_tr)
        hurdle_cb_pos_oof[va_idx] = np.clip(anchor_va * np.expm1(np.clip(reg_hurdle_cb.predict(X_va_imp), 0, 10)), 0, None)
        hurdle_cb_pos_test += np.clip(anchor_te * np.expm1(np.clip(reg_hurdle_cb.predict(X_te_imp), 0, 10)), 0, None) / 5.0

        reg_hurdle_xgb = xgb.XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            n_estimators=850, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=25,
            reg_alpha=1.0, reg_lambda=5.0, random_state=42 + fold, n_jobs=-1
        )
        reg_hurdle_xgb.fit(X_tr_raw.iloc[pos_mask], z_tr)
        hurdle_xgb_pos_oof[va_idx] = np.clip(anchor_va * np.expm1(np.clip(reg_hurdle_xgb.predict(X_va_raw), 0, 10)), 0, None)
        hurdle_xgb_pos_test += np.clip(anchor_te * np.expm1(np.clip(reg_hurdle_xgb.predict(X_te_raw), 0, 10)), 0, None) / 5.0

        # 3. Multi-Quantile Bank (LightGBM Quantile Objectives)
        for q in quantiles:
            reg_q = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=750, learning_rate=0.03,
                num_leaves=45, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                random_state=42 + fold, n_jobs=-1, verbose=-1
            )
            reg_q.fit(X_tr_raw.iloc[pos_mask], z_tr)
            q_oof_dict[q][va_idx] = np.clip(anchor_va * np.expm1(np.clip(reg_q.predict(X_va_raw), 0, 10)), 0, None)
            q_test_dict[q] += np.clip(anchor_te * np.expm1(np.clip(reg_q.predict(X_te_raw), 0, 10)), 0, None) / 5.0

        # 4. Direct Raw-Scale L1 Regressors
        reg_raw_lgb = lgb.LGBMRegressor(
            objective="mae", n_estimators=1000, learning_rate=0.03,
            num_leaves=127, max_depth=10, min_child_samples=25,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42 + fold, n_jobs=-1, verbose=-1
        )
        reg_raw_lgb.fit(X_tr_imp, y_tr)
        raw_lgb_oof[va_idx] = np.clip(reg_raw_lgb.predict(X_va_imp), 0, None)
        raw_lgb_test += np.clip(reg_raw_lgb.predict(X_te_imp), 0, None) / 5.0

        reg_raw_cb = CatBoostRegressor(
            loss_function="MAE", iterations=1200, learning_rate=0.05, depth=7,
            l2_leaf_reg=3.0, random_seed=42 + fold, thread_count=-1, verbose=0, allow_writing_files=False
        )
        reg_raw_cb.fit(X_tr_imp, y_tr)
        raw_cb_oof[va_idx] = np.clip(reg_raw_cb.predict(X_va_imp), 0, None)
        raw_cb_test += np.clip(reg_raw_cb.predict(X_te_imp), 0, None) / 5.0

        reg_raw_xgb = xgb.XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            n_estimators=900, learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42 + fold, n_jobs=-1
        )
        reg_raw_xgb.fit(X_tr_imp, y_tr)
        raw_xgb_oof[va_idx] = np.clip(reg_raw_xgb.predict(X_va_imp), 0, None)
        raw_xgb_test += np.clip(reg_raw_xgb.predict(X_te_imp), 0, None) / 5.0

        print(f"  Fold {fold}/5 completed in {time.time() - ft0:.1f}s")

    hurdle_pos_oof = 0.45 * hurdle_lgb_pos_oof + 0.35 * hurdle_cb_pos_oof + 0.20 * hurdle_xgb_pos_oof
    hurdle_pos_test = 0.45 * hurdle_lgb_pos_test + 0.35 * hurdle_cb_pos_test + 0.20 * hurdle_xgb_pos_test

    raw_ens_oof = 0.45 * raw_lgb_oof + 0.30 * raw_cb_oof + 0.25 * raw_xgb_oof
    raw_ens_test = 0.45 * raw_lgb_test + 0.30 * raw_cb_test + 0.25 * raw_xgb_test

    # Isotonic Probability Calibration & Bayes-Optimal Dynamic Quantile Prediction
    p_cal_oof = np.zeros(n, dtype=np.float64)
    p_cal_test = np.zeros(n_te, dtype=np.float64)
    bayes_quant_oof = np.zeros(n, dtype=np.float64)
    bayes_quant_test = np.zeros(n_te, dtype=np.float64)

    hurdle_3fam_oof = np.zeros(n, dtype=np.float64)
    hurdle_3fam_test = np.zeros(n_te, dtype=np.float64)
    hybrid_oof = np.zeros(n, dtype=np.float64)
    hybrid_test = np.zeros(n_te, dtype=np.float64)
    meta_oof = np.zeros(n, dtype=np.float64)
    meta_test = np.zeros(n_te, dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_key), 1):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_oof[tr_idx], y_bin[tr_idx])
        p_c_tr = iso.transform(p_oof[tr_idx])
        p_c_va = iso.transform(p_oof[va_idx])
        p_c_te = iso.transform(p_test)
        p_cal_oof[va_idx] = p_c_va
        p_cal_test += p_c_te / 5.0

        # Bayes-Optimal Quantile Hurdle per fold
        va_q_dict = {q: q_oof_dict[q][va_idx] for q in quantiles}
        te_q_dict = {q: q_test_dict[q] for q in quantiles}
        bayes_quant_oof[va_idx] = interpolate_quantile_predictions(p_c_va, quantiles, va_q_dict, tau=0.42)
        bayes_quant_test += interpolate_quantile_predictions(p_c_te, quantiles, te_q_dict, tau=0.42) / 5.0

        # Standard Hurdle
        best_th, best_s = 0.35, -1.0
        for th in np.linspace(0.22, 0.48, 53):
            s = compute_nmae(y[tr_idx], hurdle_pos_oof[tr_idx] * (p_c_tr >= th))
            if s > best_s:
                best_s, best_th = s, th
        hurdle_3fam_oof[va_idx] = hurdle_pos_oof[va_idx] * (p_c_va >= best_th)
        hurdle_3fam_test += (hurdle_pos_test * (p_c_te >= best_th)) / 5.0

        # Hybrid blend of Hurdle, Bayes-Quantile, and Raw-Scale L1
        best_w1, best_w2, best_th_h, best_sh = 0.50, 0.30, best_th, -1.0
        for w1 in [0.40, 0.50, 0.60]:
            for w2 in [0.20, 0.30, 0.40]:
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < 0:
                    continue
                tr_q_dict = {q: q_oof_dict[q][tr_idx] for q in quantiles}
                tr_bq = interpolate_quantile_predictions(p_c_tr, quantiles, tr_q_dict, tau=0.42)
                mix_tr = w1 * hurdle_pos_oof[tr_idx] * (p_c_tr >= best_th) + w2 * tr_bq + w3 * raw_ens_oof[tr_idx]
                for th2 in np.linspace(0.26, 0.44, 19):
                    s = compute_nmae(y[tr_idx], mix_tr * (p_c_tr >= th2))
                    if s > best_sh:
                        best_sh, best_w1, best_w2, best_th_h = s, w1, w2, th2

        w3_best = round(1.0 - best_w1 - best_w2, 4)
        mix_va = best_w1 * hurdle_3fam_oof[va_idx] + best_w2 * bayes_quant_oof[va_idx] + w3_best * raw_ens_oof[va_idx]
        hybrid_oof[va_idx] = mix_va * (p_c_va >= best_th_h)
        mix_te = best_w1 * hurdle_3fam_test + best_w2 * bayes_quant_test + w3_best * raw_ens_test
        hybrid_test += (mix_te * (p_c_te >= best_th_h)) / 5.0

        # Meta-Stacker features
        m_tr = np.column_stack([
            hurdle_pos_oof[tr_idx] * (p_c_tr >= best_th),
            hurdle_pos_oof[tr_idx], p_c_tr,
            q_oof_dict[0.30][tr_idx], q_oof_dict[0.50][tr_idx],
            raw_lgb_oof[tr_idx], raw_cb_oof[tr_idx], raw_xgb_oof[tr_idx],
            anchor_prev_oof[tr_idx], anchor_hist_oof[tr_idx],
        ])
        m_va = np.column_stack([
            hurdle_3fam_oof[va_idx],
            hurdle_pos_oof[va_idx], p_c_va,
            q_oof_dict[0.30][va_idx], q_oof_dict[0.50][va_idx],
            raw_lgb_oof[va_idx], raw_cb_oof[va_idx], raw_xgb_oof[va_idx],
            anchor_prev_oof[va_idx], anchor_hist_oof[va_idx],
        ])
        m_te = np.column_stack([
            hurdle_pos_test * (p_c_te >= best_th),
            hurdle_pos_test, p_c_te,
            q_test_dict[0.30], q_test_dict[0.50],
            raw_lgb_test, raw_cb_test, raw_xgb_test,
            anchor_prev_test, anchor_hist_test,
        ])
        meta_reg = lgb.LGBMRegressor(
            objective="mae", n_estimators=450, learning_rate=0.02,
            num_leaves=31, max_depth=5, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
            random_state=100 + fold, n_jobs=-1, verbose=-1
        )
        meta_reg.fit(m_tr, y[tr_idx])
        meta_oof[va_idx] = np.clip(meta_reg.predict(m_va), 0, None)
        meta_test += np.clip(meta_reg.predict(m_te), 0, None) / 5.0

    super_oof = 0.65 * hybrid_oof + 0.35 * meta_oof
    super_test = 0.65 * hybrid_test + 0.35 * meta_test

    gated_oof = np.zeros_like(y)
    gated_test = np.zeros_like(super_test)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_key), 1):
        best_th, best_s = 0.42, -1.0
        for th in np.linspace(0.35, 0.46, 23):
            s = compute_nmae(y[tr_idx], super_oof[tr_idx] * (p_cal_oof[tr_idx] >= th))
            if s > best_s:
                best_s, best_th = s, th
        gated_oof[va_idx] = super_oof[va_idx] * (p_cal_oof[va_idx] >= best_th)
        gated_test += (super_test * (p_cal_test >= best_th)) / 5.0

    nmae_score = compute_nmae(y, gated_oof)
    mae_score = mean_absolute_error(y, gated_oof)
    print(f"=== SOTA Nested OOF NMAE: {nmae_score:.10f} | MAE: {mae_score:.4f} THB ===")
    
    best_test_clean = np.round(np.clip(gated_test, 0.0, None), 2)
    ans_ids = [f"ANS{i:05d}" for i in range(1, len(test_df) + 1)]

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ans_ids, "task3": best_test_clean}).to_csv(SUBMIT_DIR / "index.csv", index=False)
    pd.DataFrame({
        "player_id": test_df["player_id"].values,
        "spending_30d": best_test_clean,
        "p_spender": np.round(p_test, 6),
        "p_calibrated": np.round(p_cal_test, 6),
        "pred_positive_median": np.round(hurdle_pos_test, 2),
        "pred_bayes_quantile": np.round(bayes_quant_test, 2),
        "pred_raw_l1_ensemble": np.round(raw_ens_test, 2),
        "pred_hybrid_hurdle": best_test_clean,
    }).to_csv(PRED_DIR / "test_predictions.csv", index=False)
    print(f"Saved elevated predictions to {SUBMIT_DIR / 'index.csv'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
