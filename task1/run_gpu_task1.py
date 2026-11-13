"""
Task 1: GPU-Accelerated (MPS/CUDA) Tabular ResNet + Fine-Split GBDT + Exclusive-Range Rule Override
Implements Deep Research recommendations for CPE 342 Anti-Cheat Pre-Filter (F2 Score):
1. Fold-safe Class-Exclusive Range detection & post-hoc probability override
2. Domain telemetry & algebraic age-recovery features (0.615 * level / level_progression_speed)
3. Adversarial Validation test-like importance weighting (matching test NaN distribution)
4. PyTorch Tabular ResNet with Residual Skip Blocks + SiLU + Cosine Annealing trained from scratch on GPU (MPS/CUDA)
5. Multi-family probability + rank ensemble with Experiment 002 fine-split GBDTs & test-like F2 threshold calibration
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

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

if platform.system() == "Darwin":
    torch.set_num_threads(1)
LGB_N_JOBS = 1 if platform.system() == "Darwin" else -1

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
SUBMIT_DIR = ROOT / "submit"
ASSETS_DIR = ROOT / "assets"

SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def best_f2_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[float, float]:
    thresholds = np.linspace(0.10, 0.35, 251)
    best_t, best_score = 0.19, -1.0
    y_b = (y_true == 1).astype(np.float64)
    w = np.ones_like(y_b) if sample_weight is None else sample_weight.astype(np.float64)
    pos_w = np.sum(w * y_b)
    for t in thresholds:
        pred = (y_prob >= t).astype(np.float64)
        tp = np.sum(w * pred * y_b)
        fp = np.sum(w * pred * (1.0 - y_b))
        denom = 4.0 * pos_w + tp + fp
        f2 = (5.0 * tp / denom) if denom > 0 else 0.0
        if f2 > best_score:
            best_score = f2
            best_t = float(t)
    return best_t, float(best_score)


def engineer_task1_features(df: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    out = df[raw_features].copy()

    # 1. Missingness telemetry signals (stripped/modified game clients drop telemetry packets)
    out["missing_count"] = out[raw_features].isna().sum(axis=1).astype(np.float32)
    for col in [
        "crosshair_placement",
        "reports_received",
        "account_age_days",
        "friend_network_size",
        "level",
        "level_progression_speed",
    ]:
        if col in out.columns:
            out[f"isna_{col}"] = out[col].isna().astype(np.float32)

    # 2. Algebraic recovery of account_age_days from level / level_progression_speed
    age_est = 0.615 * out["level"] / (out["level_progression_speed"] + 1e-6)
    age_filled = out["account_age_days"].fillna(age_est)
    out["age_filled"] = age_filled
    out["speed_multiplier"] = (
        out["level_progression_speed"] * out["account_age_days"] / (out["level"] + 1e-6)
    )
    out["friends_per_day"] = out["friend_network_size"] / (age_filled + 1.0)
    out["reports_per_day"] = out["reports_received"] / (age_filled + 1.0)
    out["level_per_day"] = out["level"] / (age_filled + 1.0)

    # 3. Biomechanical & Aimbot/Wallhack interaction ratios
    out["aim_efficiency"] = out["headshot_percentage"] / (out["reaction_time_ms"] + 1.0)
    out["lethal_velocity"] = (
        out["kill_death_ratio"] * out["accuracy_score"] / (out["reaction_time_ms"] + 1.0)
    )
    out["smurf_index"] = (out["win_rate"] * out["kill_death_ratio"]) / np.log1p(
        age_filled.clip(lower=0) + 1.0
    )
    out["aim_mechanical_ratio"] = (
        out["accuracy_score"]
        * out["headshot_percentage"]
        / (out["aiming_smoothness"] + 0.05)
    )
    out["crosshair_vs_hs"] = out["headshot_percentage"] / (
        out["crosshair_placement"] + 0.05
    )
    out["kd_per_damage"] = out["kill_death_ratio"] / (out["damage_per_round"] + 1.0)
    out["gamesense_vs_map"] = out["game_sense_score"] / (out["map_knowledge"] + 0.05)
    out["clutch_vs_survival"] = out["clutch_success_rate"] / (
        out["survival_time_avg"] + 1.0
    )

    return out.replace([np.inf, -np.inf], np.nan)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.20):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Residual Tabular Network trained from scratch on GPU (MPS/CUDA)."""

    def __init__(self, in_features: int, hidden_dim: int = 256, num_blocks: int = 3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=0.20) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


def train_gpu_tabular_resnet_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    device: torch.device,
    seed: int = 42,
    epochs: int = 18,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabularResNet(in_features=X_tr.shape[1], hidden_dim=256, num_blocks=3).to(
        device
    )
    # Asymmetric positive weight aligned with F2 recall emphasis (beta^2 = 4 -> moderate pos_weight=1.65)
    pos_weight = torch.tensor([1.65], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    n_tr = len(X_tr_t)
    bs = 1024

    best_auc = -1.0
    best_va_prob = None
    best_te_prob = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            idx = perm[i : i + bs]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        best_va_prob = torch.sigmoid(model(X_va_t)).cpu().numpy()
        best_te_prob = torch.sigmoid(model(X_te_t)).cpu().numpy()

    return best_va_prob, best_te_prob


def main() -> None:
    t0 = time.time()
    device = get_device()
    print(f"=== Task 1: GPU-Accelerated Anti-Cheat Pipeline (Device: {device}) ===")

    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    exp2_oof = pd.read_csv(PRED_DIR / "experiments/002/oof.csv")
    exp2_test = pd.read_csv(PRED_DIR / "experiments/002/test.csv")

    # Separate labeled train rows (matching exp2_oof order via player_id)
    lab = train_raw[train_raw["is_cheater"].notna()].copy().reset_index(drop=True)
    assert (lab["player_id"].values == exp2_oof["player_id"].values).all(), (
        "Order mismatch with exp2_oof"
    )

    y = lab["is_cheater"].astype(int).to_numpy()
    folds = exp2_oof["fold"].to_numpy()
    raw_features = [
        c for c in lab.columns if c not in ("player_id", "is_cheater")
    ]

    # Compute test-like importance weights (matching missingness shift between train and test)
    p_train = lab[raw_features].isna().mean().to_numpy()
    p_test = test_raw[raw_features].isna().mean().to_numpy()
    up = p_test > p_train
    is_nan = lab[raw_features].isna().to_numpy()
    log_w = np.where(
        is_nan,
        np.log(np.clip(p_test, 1e-6, 1) / np.clip(p_train, 1e-6, 1)),
        np.log(np.clip(1 - p_test, 1e-6, 1) / np.clip(1 - p_train, 1e-6, 1)),
    )[:, ~up].sum(axis=1)
    w_testlike = np.exp(log_w - log_w.max())
    w_testlike /= w_testlike.mean()

    print("1. Engineering biomechanical & telemetry features...")
    X_fe = engineer_task1_features(lab, raw_features)
    X_te_fe = engineer_task1_features(test_raw, raw_features)
    fe_cols = X_fe.columns.tolist()
    print(f"   Engineered feature count: {len(fe_cols)} (from {len(raw_features)} raw)")

    oof_lgb_fe = np.zeros(len(y), dtype=np.float64)
    test_lgb_fe = np.zeros(len(test_raw), dtype=np.float64)

    oof_gpu_mlp = np.zeros(len(y), dtype=np.float64)
    test_gpu_mlp = np.zeros(len(test_raw), dtype=np.float64)

    # Fold-safe class-exclusive range counters
    oof_legit_exclusive = np.zeros(len(y), dtype=np.int32)
    oof_cheat_exclusive = np.zeros(len(y), dtype=np.int32)
    test_legit_exclusive = np.zeros(len(test_raw), dtype=np.float64)
    test_cheat_exclusive = np.zeros(len(test_raw), dtype=np.float64)

    X_raw_np = lab[raw_features].to_numpy(dtype=np.float64)
    X_te_raw_np = test_raw[raw_features].to_numpy(dtype=np.float64)

    for k in range(5):
        trn_mask = folds != k
        val_mask = folds == k

        # A. Fold-safe Class-Exclusive Range boundaries (with 1e-4 relative margin to prevent boundary noise)
        X_tr_raw = X_raw_np[trn_mask]
        y_tr = y[trn_mask]
        lo_cheat = np.nanmin(X_tr_raw[y_tr == 1], axis=0)
        hi_cheat = np.nanmax(X_tr_raw[y_tr == 1], axis=0)
        lo_legit = np.nanmin(X_tr_raw[y_tr == 0], axis=0)
        hi_legit = np.nanmax(X_tr_raw[y_tr == 0], axis=0)

        span = np.maximum(hi_legit - lo_legit, 1e-6)
        margin = 0.001 * span

        X_va_raw = X_raw_np[val_mask]
        # Outside cheater range -> only legitimate reaches here
        legit_hits_va = np.nansum(
            (X_va_raw < (lo_cheat - margin)) | (X_va_raw > (hi_cheat + margin)), axis=1
        )
        # Outside legit range -> only cheater reaches here
        cheat_hits_va = np.nansum(
            (X_va_raw < (lo_legit - margin)) | (X_va_raw > (hi_legit + margin)), axis=1
        )
        oof_legit_exclusive[val_mask] = legit_hits_va
        oof_cheat_exclusive[val_mask] = cheat_hits_va

        legit_hits_te = np.nansum(
            (X_te_raw_np < (lo_cheat - margin)) | (X_te_raw_np > (hi_cheat + margin)),
            axis=1,
        )
        cheat_hits_te = np.nansum(
            (X_te_raw_np < (lo_legit - margin)) | (X_te_raw_np > (hi_legit + margin)),
            axis=1,
        )
        test_legit_exclusive += legit_hits_te / 5.0
        test_cheat_exclusive += cheat_hits_te / 5.0

        # B. Feature-Engineered Fine-Split LightGBM (max_bin=4095)
        X_tr_df = X_fe.iloc[trn_mask].copy()
        X_va_df = X_fe.iloc[val_mask].copy()
        X_te_df = X_te_fe.copy()
        X_tr_df["legit_hits"] = np.nansum(
            (X_tr_raw < (lo_cheat - margin)) | (X_tr_raw > (hi_cheat + margin)), axis=1
        )
        X_tr_df["cheat_hits"] = np.nansum(
            (X_tr_raw < (lo_legit - margin)) | (X_tr_raw > (hi_legit + margin)), axis=1
        )
        X_va_df["legit_hits"] = legit_hits_va
        X_va_df["cheat_hits"] = cheat_hits_va
        X_te_df["legit_hits"] = legit_hits_te
        X_te_df["cheat_hits"] = cheat_hits_te

        lgb_model = lgb.LGBMClassifier(
            n_estimators=550,
            learning_rate=0.045,
            num_leaves=63,
            max_bin=511,
            min_child_samples=45,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.5,
            reg_lambda=5.0,
            random_state=SEED + k,
            n_jobs=LGB_N_JOBS,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=[(X_va_df, y[val_mask])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        oof_lgb_fe[val_mask] = lgb_model.predict_proba(X_va_df)[:, 1]
        test_lgb_fe += lgb_model.predict_proba(X_te_df)[:, 1] / 5.0

        # C. PyTorch GPU TabularResNet (trained from scratch on MPS/CUDA)
        med = X_tr_df.median()
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_df.fillna(med).to_numpy(dtype=np.float32))
        X_va_sc = scaler.transform(X_va_df.fillna(med).to_numpy(dtype=np.float32))
        X_te_sc = scaler.transform(X_te_df.fillna(med).to_numpy(dtype=np.float32))

        va_mlp, te_mlp = train_gpu_tabular_resnet_fold(
            X_tr_sc,
            y_tr,
            X_va_sc,
            y[val_mask],
            X_te_sc,
            device=device,
            seed=SEED + k,
            epochs=16,
            batch_size=512,
        )
        oof_gpu_mlp[val_mask] = va_mlp
        test_gpu_mlp += te_mlp / 5.0

        print(
            f"   Fold {k+1}/5 | LGB_FE AUC: {roc_auc_score(y[val_mask], oof_lgb_fe[val_mask]):.5f} | "
            f"GPU_ResNet AUC: {roc_auc_score(y[val_mask], oof_gpu_mlp[val_mask]):.5f}"
        )

    # Compare baseline Exp 002 Blend vs Multi-Family GPU + FE Blend
    oof_exp2 = exp2_oof["oof_Blend"].to_numpy()
    test_exp2 = exp2_test["proba_Blend"].to_numpy()

    base_t, base_f2 = best_f2_threshold(y, oof_exp2)
    base_tl_t, base_tl_f2 = best_f2_threshold(y, oof_exp2, w_testlike)
    print(
        f"\nBaseline Exp002 Blend | OOF AUC: {roc_auc_score(y, oof_exp2):.5f} | "
        f"OOF F2: {base_f2:.5f} (tau={base_t:.4f}) | Weighted Test-Like F2: {base_tl_f2:.5f}"
    )

    # Grid search optimal ensemble weights + exclusive-range dampening/boosting on OOF
    best_cfg = (1.0, 0.0, 0.0, 1.0, 0.0, base_t)
    best_oof_f2 = base_f2
    best_oof_prob = oof_exp2.copy()

    for w_exp2 in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
        for w_lgb_fe in [0.00, 0.05, 0.10, 0.15, 0.20]:
            w_mlp = round(1.0 - w_exp2 - w_lgb_fe, 4)
            if w_mlp < 0 or w_mlp > 0.20:
                continue
            raw_blend = (
                w_exp2 * oof_exp2 + w_lgb_fe * oof_lgb_fe + w_mlp * oof_gpu_mlp
            )
            for legit_damp in [1.0, 0.50, 0.25]:
                for cheat_boost in [0.0, 0.25, 0.50]:
                    adj = raw_blend.copy()
                    # When a sample falls strictly into a legit-only range and has 0 cheater hits
                    mask_pure_legit = (oof_legit_exclusive >= 1) & (
                        oof_cheat_exclusive == 0
                    )
                    mask_pure_cheat = (oof_cheat_exclusive >= 1) & (
                        oof_legit_exclusive == 0
                    )
                    adj[mask_pure_legit] *= legit_damp
                    adj[mask_pure_cheat] = np.maximum(
                        adj[mask_pure_cheat], cheat_boost
                    )
                    tau, score_f2 = best_f2_threshold(y, adj)
                    if score_f2 > best_oof_f2:
                        best_oof_f2 = score_f2
                        best_cfg = (w_exp2, w_lgb_fe, w_mlp, legit_damp, cheat_boost, tau)
                        best_oof_prob = adj.copy()

    w_exp2, w_lgb_fe, w_mlp, legit_damp, cheat_boost, opt_tau = best_cfg
    tl_tau, best_tl_f2 = best_f2_threshold(y, best_oof_prob, w_testlike)
    final_tau = 0.5 * opt_tau + 0.5 * tl_tau

    print(
        f"Upgraded GPU+GBDT+Rules | Weights=({w_exp2:.2f} Exp2, {w_lgb_fe:.2f} LGB_FE, {w_mlp:.2f} GPU_ResNet) "
        f"| Rules=(legit*{legit_damp}, cheat>={cheat_boost})"
    )
    print(
        f"Upgraded OOF AUC: {roc_auc_score(y, best_oof_prob):.5f} | "
        f"OOF F2: {best_oof_f2:.5f} (+{best_oof_f2 - base_f2:.5f}) | "
        f"Weighted Test-Like F2: {best_tl_f2:.5f} | Final tau={final_tau:.4f}"
    )

    # Apply to Test Set
    test_prob = (
        w_exp2 * test_exp2 + w_lgb_fe * test_lgb_fe + w_mlp * test_gpu_mlp
    )
    mask_te_legit = (test_legit_exclusive >= 0.8) & (test_cheat_exclusive == 0)
    mask_te_cheat = (test_cheat_exclusive >= 0.8) & (test_legit_exclusive == 0)
    test_prob[mask_te_legit] *= legit_damp
    test_prob[mask_te_cheat] = np.maximum(test_prob[mask_te_cheat], cheat_boost)

    test_preds = (test_prob >= final_tau).astype(int)

    sample_sub = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame({"id": sample_sub["id"], "task1": test_preds})
    sub_df.to_csv(SUBMIT_DIR / "index.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "player_id": test_raw["player_id"],
            "is_cheater": test_preds,
            "prob_cheater": test_prob,
            "prob_exp2_gbdt": test_exp2,
            "prob_lgb_fe": test_lgb_fe,
            "prob_gpu_resnet": test_gpu_mlp,
        }
    )
    pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)

    pos_cnt = int(test_preds.sum())
    print(
        f"Saved Task 1 predictions to {SUBMIT_DIR / 'index.csv'}: "
        f"1={pos_cnt:,} ({pos_cnt/len(test_preds)*100:.1f}%), 0={len(test_preds)-pos_cnt:,} "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()

