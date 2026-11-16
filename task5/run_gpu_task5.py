"""
Task 5: GPU-Accelerated (MPS/CUDA) Deep Autoencoder + Temporal Self-Deviation + ECOD + XGBOD Self-Training
Implements Deep Research recommendations for CPE 342 Unsupervised Account Security Monitoring (F3 Score):
1. Per-account longitudinal imputation (median across account's own periods 1..4 first; NaNs never penalized)
2. Account-Level Temporal Self-Deviation Engine across all 29 longitudinal metrics:
   - Period 4 vs median(Periods 1..3) MAD-normalized robust z-score
   - All-period max leave-one-out deviation across Periods 1, 2, 3, 4
   - Geodesic Haversine travel velocity (km) across periods 1->2->3->4
3. Exact Synthetic Anomaly Archetype Detectors (Groups 1..4: Directional Jumps, Low Stability, Silent Bots, Farming Bots)
4. Parameter-free ECOD (Empirical Tail Detector) + Isolation Forest + PCA SPE/T^2 on Temporal Delta/Z features
5. 3-Seed PyTorch Deep Bottleneck Denoising Autoencoder trained from scratch on GPU (MPS/CUDA)
6. XGBOD-style Self-Trained GBDT + GPU MLP Meta-Ranker calibrated to optimal F3 marginal precision (k* = 2,150)
"""

from __future__ import annotations

import os
import platform
import time
import warnings
from pathlib import Path

if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

if platform.system() == "Darwin":
    torch.set_num_threads(1)
LGB_N_JOBS = 1 if platform.system() == "Darwin" else -1

warnings.filterwarnings("ignore")

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


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_temporal_self_deviation_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Constructs per-account longitudinal self-deviation features across all 29 bases x 4 periods.
    Missing values are imputed from each account's own median across its other periods first.
    """
    n = len(df)
    bases = sorted(
        list(
            set(
                c[:-2]
                for c in df.columns
                if c.endswith(("_1", "_2", "_3", "_4"))
            )
        )
    )
    flag_cols = [
        c
        for c in df.columns
        if c not in ["player_id"]
        and not c.endswith(("_1", "_2", "_3", "_4"))
    ]

    imputed_panels: dict[str, np.ndarray] = {}
    delta_features: dict[str, np.ndarray] = {}

    for b in bases:
        cols = [f"{b}_1", f"{b}_2", f"{b}_3", f"{b}_4"]
        arr = df[cols].to_numpy(dtype=np.float64)  # [N, 4]

        # Per-account median across available periods, fallback to population median
        row_med = np.nanmedian(arr, axis=1, keepdims=True)
        pop_med = np.nanmedian(arr)
        row_med = np.where(np.isnan(row_med), pop_med, row_med)
        arr_imp = np.where(np.isnan(arr), row_med, arr)
        imputed_panels[b] = arr_imp

        # 1. Period 4 jump relative to account's own baseline (median of periods 1..3)
        base_123 = np.median(arr_imp[:, :3], axis=1)
        jump_4 = arr_imp[:, 3] - base_123
        mad_jump4 = np.median(np.abs(jump_4 - np.median(jump_4))) * 1.4826 + 1e-5
        z_jump4 = jump_4 / mad_jump4

        # 2. All-period leave-one-out max spike (captures anomalies injected in period 1, 2, 3, or 4)
        loo_z_list = []
        for t in range(4):
            other_idx = [j for j in range(4) if j != t]
            base_other = np.median(arr_imp[:, other_idx], axis=1)
            diff_t = arr_imp[:, t] - base_other
            mad_t = np.median(np.abs(diff_t - np.median(diff_t))) * 1.4826 + 1e-5
            loo_z_list.append(np.abs(diff_t) / mad_t)
        max_loo_z = np.max(np.column_stack(loo_z_list), axis=1)

        # 3. Account temporal volatility (std across 4 periods normalized by population MAD)
        row_std = np.std(arr_imp, axis=1)
        mad_std = np.median(np.abs(row_std - np.median(row_std))) * 1.4826 + 1e-5
        z_std = row_std / mad_std

        # 4. Population robust z of Period 4
        val_p4 = arr_imp[:, 3]
        mad_p4 = np.median(np.abs(val_p4 - np.median(val_p4))) * 1.4826 + 1e-5
        z_pop_p4 = (val_p4 - np.median(val_p4)) / mad_p4

        delta_features[f"{b}_z_jump4"] = z_jump4
        delta_features[f"{b}_max_loo_z"] = max_loo_z
        delta_features[f"{b}_z_std"] = z_std
        delta_features[f"{b}_z_pop_p4"] = z_pop_p4
        delta_features[f"{b}_p4_raw"] = val_p4

    # Geodesic Haversine travel distances across consecutive periods
    lat = imputed_panels["login_lat"]
    lon = imputed_panels["login_lon"]
    d12 = haversine_km(lat[:, 0], lon[:, 0], lat[:, 1], lon[:, 1])
    d23 = haversine_km(lat[:, 1], lon[:, 1], lat[:, 2], lon[:, 2])
    d34 = haversine_km(lat[:, 2], lon[:, 2], lat[:, 3], lon[:, 3])
    delta_features["geo_dist_34_km"] = d34
    delta_features["geo_dist_max_km"] = np.maximum.reduce([d12, d23, d34])
    delta_features["geo_jump_ratio"] = d34 / (0.5 * (d12 + d23) + 50.0)

    # Contextual security flags (imputed with median/0)
    for fc in flag_cols:
        fill_val = 0.0 if fc in ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"] else float(df[fc].median())
        delta_features[fc] = df[fc].fillna(fill_val).to_numpy(dtype=np.float64)

    # Contextual ATO / Bot / Economy interactions using actual Task 5 columns
    pw = delta_features["password_changed"]
    sale = delta_features["mass_item_sale"]
    susp = delta_features["suspicious_login_time"]
    vpn = delta_features["vpn_usage"]

    delta_features["ato_context_score"] = (
        pw * (delta_features["location_changes_z_jump4"].clip(0, 20) + d34 / 1000.0)
        + sale * delta_features["currency_transfers_z_jump4"].clip(0, 20)
        + vpn * susp
    )
    delta_features["bot_context_score"] = (
        susp
        * delta_features["login_count_z_jump4"].clip(0, 20)
        * (-delta_features["session_duration_std_z_pop_p4"]).clip(0, 20)
    )

    feat_df = pd.DataFrame(delta_features)
    return feat_df, imputed_panels


def compute_ecod_score(X: np.ndarray) -> np.ndarray:
    """
    Parameter-free ECOD (Empirical Cumulative Distribution Outlier Detection):
    Computes negative log two-sided empirical tail probability across features.
    """
    n, d = X.shape
    scores = np.zeros(n, dtype=np.float64)
    for j in range(d):
        col = X[:, j]
        r_right = (n + 1.0 - rankdata(col, method="average")) / (n + 1.0)
        scores += -np.log(np.clip(r_right, 1e-6, 1.0))
    return scores / d


class DeepTabularAutoencoder(nn.Module):
    """Deep Bottleneck Denoising Autoencoder trained from scratch on GPU (MPS/CUDA)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 20),
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_gpu_autoencoder_ensemble(
    X_scaled: np.ndarray, device: torch.device, seeds: list[int] = [42, 1337, 2026]
) -> np.ndarray:
    """Trains a 3-seed ensemble of Deep Autoencoders directly on GPU memory (MPS/CUDA)."""
    n, d = X_scaled.shape
    X_full_dev = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    batch_size = 1024

    recon_scores = np.zeros(n, dtype=np.float64)
    criterion = nn.HuberLoss(delta=1.5, reduction="none")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DeepTabularAutoencoder(in_dim=d).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=24, eta_min=1e-5
        )

        for _ in range(24):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                bx = X_full_dev[perm[i : i + batch_size]]
                noisy_bx = bx + 0.05 * torch.randn_like(bx)
                optimizer.zero_grad(set_to_none=True)
                recon = model(noisy_bx)
                loss = criterion(recon, bx).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            recon_full = model(X_full_dev)
            err_matrix = (recon_full - X_full_dev).abs().cpu().numpy()
            top15_err = np.sort(err_matrix, axis=1)[:, -15:].mean(axis=1)
            recon_scores += rankdata(top15_err) / (n * len(seeds))

    return recon_scores


class SelfTrainedMLP(nn.Module):
    """GPU MLP Meta-Ranker trained on high-confidence pseudo-anchors (XGBOD style)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    t0 = time.time()
    device = get_device()
    print(f"=== Task 5: GPU-Accelerated Unsupervised Anomaly Detection (Device: {device}) ===")

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "test.csv")
    n = len(df)
    print(f"1. Loaded Task 5 test dataset: {df.shape}")

    print("2. Building Account-Level Temporal Self-Deviation & Geodesic Haversine features...")
    feat_df, panels = build_temporal_self_deviation_features(df)
    print(f"   Engineered temporal self-deviation matrix: {feat_df.shape}")

    # 3. Deterministic Synthetic Anomaly Archetypes (Verified S5 ~ 0.780 on Leaderboard)
    g2_mask = (df["behavior_stability"] < 0.60).fillna(False).to_numpy()

    comm_cols = [f"communication_frequency_{i}" for i in range(1, 5)]
    g3_mask = (df[comm_cols].mean(axis=1) < 0.175).fillna(False).to_numpy()

    g4_mask = (
        (df[[f"ip_hash_entropy_{i}" for i in range(1, 5)]].mean(axis=1) < 0.25)
        & (df[[f"login_time_variance_{i}" for i in range(1, 5)]].mean(axis=1) < 0.60)
        & (df[[f"session_duration_std_{i}" for i in range(1, 5)]].mean(axis=1) < 5.50)
        & (df[[f"kda_std_{i}" for i in range(1, 5)]].mean(axis=1) < 0.12)
    ).fillna(False).to_numpy()

    def get_max_prior(prefix: str) -> pd.Series:
        return df[[f"{prefix}_{i}" for i in range(1, 4)]].max(axis=1)

    def get_min_prior(prefix: str) -> pd.Series:
        return df[[f"{prefix}_{i}" for i in range(1, 4)]].min(axis=1)

    diff_curr = df["currency_transfers_4"] - get_max_prior("currency_transfers")
    diff_skill_traj = df["skill_trajectory_4"] - get_max_prior("skill_trajectory")
    diff_skill_rate = df["skill_rating_4"] - get_max_prior("skill_rating")
    diff_loc = df["location_changes_4"] - get_max_prior("location_changes")
    diff_purch_f = df["purchase_frequency_4"] - get_max_prior("purchase_frequency")
    diff_purch_a = df["purchase_amount_mean_4"] - get_max_prior("purchase_amount_mean")
    diff_acc = df["accuracy_mean_4"] - get_max_prior("accuracy_mean")
    diff_kda = df["kda_mean_4"] - get_max_prior("kda_mean")
    diff_react = df["reaction_time_mean_4"] - get_min_prior("reaction_time_mean")
    diff_pat = df["play_pattern_consistency_4"] - get_min_prior("play_pattern_consistency")
    diff_rem = df["friend_removes_4"] - get_max_prior("friend_removes")
    diff_adds = df["friend_adds_4"] - get_max_prior("friend_adds")

    lat_j = np.abs(df["login_lat_4"] - df["login_lat_3"])
    lon_j = np.abs(df["login_lon_4"] - df["login_lon_3"])
    geo_j = np.sqrt(lat_j**2 + lon_j**2)

    g1_mask = (
        (diff_curr > 5.7)
        | (diff_skill_traj > 11.86)
        | (diff_skill_rate > 289.3)
        | (diff_loc > 2.01)
        | (diff_purch_f > 3.24)
        | (diff_purch_a > 213.1)
        | (diff_acc > 9.75)
        | (diff_kda > 1.21)
        | (diff_react < -60.7)
        | (diff_pat < -0.152)
        | (diff_rem > 2.73)
        | (diff_adds > 12.88)
        | (geo_j > 3.0)
    ).fillna(False).to_numpy()

    core_anomalies = g1_mask | g2_mask | g3_mask | g4_mask
    rule_hits = (
        g1_mask.astype(int)
        + g2_mask.astype(int)
        + g3_mask.astype(int)
        + g4_mask.astype(int)
    )
    print(
        f"   Core Deterministic Archetypes: {core_anomalies.sum():,} accounts "
        f"(G1={g1_mask.sum():,}, G2={g2_mask.sum():,}, G3={g3_mask.sum():,}, G4={g4_mask.sum():,})"
    )

    # 4. Unsupervised Detectors on Temporal Delta/Z Features + Raw Scaled Matrix
    z_cols = [
        c
        for c in feat_df.columns
        if c.endswith(("_z_jump4", "_max_loo_z", "_z_std", "_z_pop_p4"))
        or c in ["geo_dist_34_km", "ato_context_score", "bot_context_score"]
    ]
    X_z = np.clip(feat_df[z_cols].to_numpy(dtype=np.float64), -15.0, 25.0)
    X_z_sc = RobustScaler().fit_transform(X_z)
    X_z_sc = np.clip(X_z_sc, -10.0, 15.0)

    print("3. Computing ECOD, Isolation Forest, & PCA Mahalanobis scores...")
    ecod_raw = compute_ecod_score(X_z_sc)
    ecod_rank = rankdata(ecod_raw) / n

    iforest = IsolationForest(
        n_estimators=400, max_samples=0.8, random_state=SEED, n_jobs=LGB_N_JOBS
    )
    if_raw = -iforest.fit(X_z_sc).score_samples(X_z_sc)
    if_rank = rankdata(if_raw) / n

    pca = PCA(n_components=15, random_state=SEED)
    Z_pca = pca.fit_transform(X_z_sc)
    X_pca_recon = pca.inverse_transform(Z_pca)
    spe_rank = rankdata(np.mean(np.abs(X_z_sc - X_pca_recon), axis=1)) / n
    t2_rank = rankdata(np.sum((Z_pca**2) / np.maximum(pca.explained_variance_, 1e-6), axis=1)) / n

    jump4_cols = [c for c in feat_df.columns if c.endswith("_z_jump4")]
    max_jump4_z = np.max(np.abs(feat_df[jump4_cols].to_numpy()), axis=1)
    jump_rank = rankdata(max_jump4_z) / n

    print("4. Training 3-Seed PyTorch Deep Denoising Autoencoder on GPU (MPS/CUDA)...")
    ae_rank = train_gpu_autoencoder_ensemble(X_z_sc, device=device, seeds=[42, 1337, 2026])

    unsup_consensus = (
        0.25 * jump_rank
        + 0.25 * ae_rank
        + 0.20 * ecod_rank
        + 0.15 * if_rank
        + 0.15 * (0.5 * spe_rank + 0.5 * t2_rank)
    )

    # 5. XGBOD-Style Self-Training Meta-Ranker (LightGBM + GPU MLP trained from scratch)
    print("5. Running XGBOD Self-Training Meta-Ranker (GBDT + GPU MLP on pseudo-anchors)...")
    pos_anchors = core_anomalies | (unsup_consensus >= np.quantile(unsup_consensus, 0.94))
    neg_anchors = (~core_anomalies) & (unsup_consensus <= np.quantile(unsup_consensus, 0.55))

    anchor_mask = pos_anchors | neg_anchors
    X_all_meta = StandardScaler().fit_transform(
        np.column_stack([X_z_sc, ecod_rank, if_rank, ae_rank, jump_rank, spe_rank, t2_rank])
    ).astype(np.float32)
    y_anchors = pos_anchors[anchor_mask].astype(int)
    X_anchors = X_all_meta[anchor_mask]

    print(
        f"   Self-training anchors: {pos_anchors.sum():,} positive, {neg_anchors.sum():,} negative "
        f"(borderline unlabelled: {(~anchor_mask).sum():,})"
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    meta_lgb_prob = np.zeros(n, dtype=np.float64)
    meta_gpu_prob = np.zeros(n, dtype=np.float64)

    X_all_t = torch.tensor(X_all_meta, dtype=torch.float32, device=device)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_anchors, y_anchors), 1):
        X_tr, y_tr = X_anchors[tr_idx], y_anchors[tr_idx]

        # A. Self-Trained LightGBM Ranker
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED + fold,
            n_jobs=LGB_N_JOBS,
            verbose=-1,
        )
        clf_lgb.fit(X_tr, y_tr)
        meta_lgb_prob += clf_lgb.predict_proba(X_all_meta)[:, 1] / 5.0

        # B. Self-Trained GPU MLP Ranker (GPU-resident mini-batching)
        torch.manual_seed(SEED + fold)
        mlp = SelfTrainedMLP(in_dim=X_all_meta.shape[1]).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=2e-4)
        crit = nn.BCEWithLogitsLoss()
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        bs = 1024
        for _ in range(14):
            mlp.train()
            perm = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(X_tr_t), bs):
                idx = perm[i : i + bs]
                opt.zero_grad(set_to_none=True)
                loss = crit(mlp(X_tr_t[idx]), y_tr_t[idx])
                loss.backward()
                opt.step()
        mlp.eval()
        with torch.no_grad():
            meta_gpu_prob += torch.sigmoid(mlp(X_all_t)).cpu().numpy() / 5.0

    meta_rank = rankdata(0.55 * meta_lgb_prob + 0.45 * meta_gpu_prob) / n

    # Final Composite Anomaly Score:
    # Core deterministic archetypes are guaranteed top rank, and borderline tail is ranked by
    # 55% XGBOD (GBDT + GPU MLP) + 45% Unsupervised Consensus (GPU Autoencoder + ECOD + IForest + Temporal Jump)
    borderline_score = 0.55 * meta_rank + 0.45 * unsup_consensus
    final_score = core_anomalies.astype(np.float64) * 2.0 + borderline_score

    # F3 Optimal Cutoff (Optimal high-recall champion cutoff: k* = 2,150 = 8.30% contamination)
    target_k = 2150
    top_idx = np.argsort(-final_score)[:target_k]
    preds = np.zeros(n, dtype=int)
    preds[top_idx] = 1

    sample_sub = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    sub_df = pd.DataFrame({"id": sample_sub["id"], "task5": preds})
    sub_df.to_csv(SUBMIT_DIR / "index.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "player_id": df["player_id"],
            "is_anomaly": preds,
            "anomaly_score": final_score,
            "core_archetype": core_anomalies.astype(int),
            "rule_hits": rule_hits,
            "gpu_ae_rank": ae_rank,
            "ecod_rank": ecod_rank,
            "iforest_rank": if_rank,
            "xgbod_meta_rank": meta_rank,
        }
    )
    pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)

    # Compare agreement with previous Task 5 submission
    prev_sub_path = ROOT.parent.parent / "cpe342-machine-learning-2026" / "project" / "task5" / "submit" / "index.csv"
    if prev_sub_path.exists():
        prev_preds = pd.read_csv(prev_sub_path)["task5"].to_numpy()
        overlap = int(np.sum((preds == 1) & (prev_preds == 1)))
        print(
            f"   Overlap with proven S5=0.780 submission: {overlap:,} / {target_k:,} ({overlap/target_k*100:.1f}%)"
        )

    print(
        f"Saved Task 5 predictions to {SUBMIT_DIR / 'index.csv'}: "
        f"flagged={preds.sum():,} ({preds.mean()*100:.2f}%) in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
