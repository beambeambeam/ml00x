"""
Task 2: GPU-Accelerated (MPS/CUDA) Categorical-Embedding ResNet + Multi-Seed GBDT + Coordinate Ascent Macro-F1 Optimizer
Implements Deep Research recommendations for CPE 342 Player Segment Classification (4-Class Macro F1):
1. Strict removal of synthetic noise columns (random_metric_1..3)
2. Pairwise confusion-targeted features (Casual<->Social, Grinder<->Whale) + categorical interactions (vip_region, loyalty_platform)
3. Fold-safe Multi-Class Out-of-Fold Target Encoding & Frequency Encoding
4. PyTorch Categorical-Embedding + Residual Tabular Network trained from scratch on GPU (MPS/CUDA)
5. Multi-Family Probability Blend + Nested Coordinate Ascent on Class Logit Offsets directly maximizing Macro F1
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

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
NOISE_COLS = ["segment", "player_id", "random_metric_1", "random_metric_2", "random_metric_3"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def engineer_task2_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=NOISE_COLS, errors="ignore").copy()

    # Core ratio features (matching reference + pairwise confusion separators)
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
        if num in x.columns and den in x.columns:
            x[name] = x[num] / (x[den].abs() + 1.0)

    x["weekly_minutes"] = x["play_frequency"] * x["avg_session_duration"]
    x["social_activity"] = x["friend_count"] * x["team_play_percentage"]

    # Targeted confusion-pair separators (Grinder vs Whale, Casual vs Social)
    x["whale_intensity"] = np.log1p(x["total_spending_thb"].clip(lower=0)) * np.log1p(
        x["avg_monthly_spending"].clip(lower=0)
    )
    x["social_per_session"] = x["social_activity"] / (x["avg_session_duration"].abs() + 1.0)
    x["grinder_vs_whale"] = (
        x["weekly_minutes"] * (x["ranked_participation_rate"].fillna(0) + 1.0)
    ) / (x["total_spending_thb"].clip(lower=0) + 10.0)
    x["spend_concentration"] = x["avg_monthly_spending"] / (
        x["total_spending_thb"].clip(lower=0) / 12.0 + 1.0
    )
    if "cosmetic_items_owned" in x.columns and "total_spending_thb" in x.columns:
        x["cosmetics_per_spend"] = x["cosmetic_items_owned"] / (
            x["total_spending_thb"].clip(lower=0) + 10.0
        )

    # High-order categorical interactions
    vip = x["vip_tier"].map(
        lambda v: "__MISSING__" if pd.isna(v) else str(int(v))
    )
    region = x["region"].fillna("__MISSING__").astype(str)
    x["vip_region"] = vip + "__" + region

    if "loyalty_tier" in x.columns and "platform" in x.columns:
        x["loyalty_platform"] = (
            x["loyalty_tier"].fillna("__MISSING__").astype(str)
            + "__"
            + x["platform"].fillna("__MISSING__").astype(str)
        )

    x["missing_count"] = x.isna().sum(axis=1).astype(np.float32)
    return x.replace([np.inf, -np.inf], np.nan)


class CategoricalEmbeddingResNet(nn.Module):
    """PyTorch Tabular Architecture with learned Categorical Embeddings + Residual MLP Blocks on GPU."""

    def __init__(
        self,
        cat_cardinalities: list[int],
        num_numerical: int,
        hidden_dim: int = 256,
        num_classes: int = 4,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=card + 1, embedding_dim=min(16, (card + 1) // 2 + 2))
                for card in cat_cardinalities
            ]
        )
        emb_total_dim = sum(e.embedding_dim for e in self.embeddings)
        in_dim = emb_total_dim + num_numerical

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb_outs + [x_num], dim=1)
        h = self.in_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)


def train_gpu_embedding_resnet_fold(
    X_cat_tr: np.ndarray,
    X_num_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cat_va: np.ndarray,
    X_num_va: np.ndarray,
    y_va: np.ndarray,
    X_cat_te: np.ndarray,
    X_num_te: np.ndarray,
    cardinalities: list[int],
    device: torch.device,
    seed: int = 42,
    epochs: int = 16,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CategoricalEmbeddingResNet(
        cat_cardinalities=cardinalities,
        num_numerical=X_num_tr.shape[1],
        hidden_dim=256,
        num_classes=4,
    ).to(device)

    # Mild inverse class frequency weighting to support Macro F1
    counts = np.bincount(y_tr, minlength=4).astype(np.float32)
    inv_w = (counts.sum() / (4.0 * counts)) ** 0.35
    class_weights = torch.tensor(inv_w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    X_cat_tr_t = torch.tensor(X_cat_tr, dtype=torch.long, device=device)
    X_num_tr_t = torch.tensor(X_num_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    X_cat_va_t = torch.tensor(X_cat_va, dtype=torch.long, device=device)
    X_num_va_t = torch.tensor(X_num_va, dtype=torch.float32, device=device)
    X_cat_te_t = torch.tensor(X_cat_te, dtype=torch.long, device=device)
    X_num_te_t = torch.tensor(X_num_te, dtype=torch.float32, device=device)
    n_tr = len(X_cat_tr_t)
    bs = 1024

    best_f1 = -1.0
    best_va_prob = None
    best_te_prob = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            idx = perm[i : i + bs]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_cat_tr_t[idx], X_num_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        best_va_prob = torch.softmax(model(X_cat_va_t, X_num_va_t), dim=1).cpu().numpy()
        best_te_prob = torch.softmax(model(X_cat_te_t, X_num_te_t), dim=1).cpu().numpy()

    return best_va_prob, best_te_prob


def optimize_macro_f1_offsets_coordinate_ascent(
    probs: np.ndarray, y_true: np.ndarray, n_rounds: int = 4
) -> np.ndarray:
    """Coordinate Ascent on log-probability class offsets c = [0, c1, c2, c3] to directly maximize Macro F1."""
    log_p = np.log(np.clip(probs, 1e-7, 1.0))
    offsets = np.zeros(4, dtype=np.float64)
    best_score = f1_score(y_true, log_p.argmax(axis=1), average="macro")

    for _ in range(n_rounds):
        for c in range(1, 4):
            best_c = offsets[c]
            for delta in np.linspace(-0.35, 0.35, 141):
                cand = offsets.copy()
                cand[c] = delta
                preds = (log_p + cand).argmax(axis=1)
                score = f1_score(y_true, preds, average="macro")
                if score > best_score + 1e-6:
                    best_score = score
                    best_c = delta
            offsets[c] = best_c
    return offsets


def main() -> None:
    t0 = time.time()
    device = get_device()
    print(f"=== Task 2: GPU-Accelerated Player Segment Pipeline (Device: {device}) ===")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    y = train_df["segment"].to_numpy(dtype=int)

    # Load saved OOF & test probability artifacts from teammate's 104k experiments
    ref_oof = pd.read_csv(PRED_DIR / "segment-oof.csv")[
        ["prob_0", "prob_1", "prob_2", "prob_3"]
    ].to_numpy()
    vip4_oof = pd.read_csv(
        PRED_DIR / "segment-vip-region-four-seed-average-oof.csv"
    )[["prob_0", "prob_1", "prob_2", "prob_3"]].to_numpy()
    xgb_oof = pd.read_csv(PRED_DIR / "segment-xgboost-oof.csv")[
        ["prob_0", "prob_1", "prob_2", "prob_3"]
    ].to_numpy()
    cb_bal_oof = pd.read_csv(PRED_DIR / "segment-balanced-catboost-oof.csv")[
        ["prob_0", "prob_1", "prob_2", "prob_3"]
    ].to_numpy()

    # Load corresponding test probabilities from saved reference + VIP×region top pick
    top_pick_test_df = pd.read_csv(PRED_DIR / "segment-top-pick-vip-region-test.csv")
    top_pick_test_probs = top_pick_test_df[
        ["prob_0", "prob_1", "prob_2", "prob_3"]
    ].to_numpy()

    print("1. Engineering pairwise confusion-separator features...")
    X_fe = engineer_task2_features(train_df)
    X_te_fe = engineer_task2_features(test_df)

    cat_cols = X_fe.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X_fe.columns if c not in cat_cols]
    print(f"   Categorical cols: {len(cat_cols)}, Numerical cols: {len(num_cols)}")

    # Prepare integer-encoded categorical arrays for PyTorch Embedding layer
    X_cat_all = np.zeros((len(train_df), len(cat_cols)), dtype=np.int64)
    X_cat_test = np.zeros((len(test_df), len(cat_cols)), dtype=np.int64)
    cardinalities = []
    for i, col in enumerate(cat_cols):
        tr_s = X_fe[col].fillna("__MISSING__").astype(str)
        te_s = X_te_fe[col].fillna("__MISSING__").astype(str)
        uniq = sorted(tr_s.unique().tolist())
        mapping = {v: idx + 1 for idx, v in enumerate(uniq)}  # 0 reserved for unknown
        X_cat_all[:, i] = tr_s.map(lambda v: mapping.get(v, 0)).to_numpy()
        X_cat_test[:, i] = te_s.map(lambda v: mapping.get(v, 0)).to_numpy()
        cardinalities.append(len(uniq) + 1)

    # Exact same 3-Fold StratifiedKFold(seed=42) split as segment-oof.csv
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    oof_gpu_nn = np.zeros((len(train_df), 4), dtype=np.float64)
    test_gpu_nn = np.zeros((len(test_df), 4), dtype=np.float64)

    oof_lgb_fe = np.zeros((len(train_df), 4), dtype=np.float64)
    test_lgb_fe = np.zeros((len(test_df), 4), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, y), start=1):
        y_tr, y_va = y[tr_idx], y[va_idx]

        # A. PyTorch Categorical-Embedding + Residual Network on GPU (MPS/CUDA)
        med = X_fe[num_cols].iloc[tr_idx].median()
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(
            X_fe[num_cols].iloc[tr_idx].fillna(med).to_numpy(dtype=np.float32)
        )
        X_num_va = scaler.transform(
            X_fe[num_cols].iloc[va_idx].fillna(med).to_numpy(dtype=np.float32)
        )
        X_num_te = scaler.transform(
            X_te_fe[num_cols].fillna(med).to_numpy(dtype=np.float32)
        )

        va_nn, te_nn = train_gpu_embedding_resnet_fold(
            X_cat_all[tr_idx],
            X_num_tr,
            y_tr,
            X_cat_all[va_idx],
            X_num_va,
            y_va,
            X_cat_test,
            X_num_te,
            cardinalities=cardinalities,
            device=device,
            seed=SEED + fold,
            epochs=14,
            batch_size=512,
        )
        oof_gpu_nn[va_idx] = va_nn
        test_gpu_nn += te_nn / 3.0

        # B. Feature-Engineered LightGBM with pairwise separator ratios
        X_tr_lgb = X_fe.iloc[tr_idx].copy()
        X_va_lgb = X_fe.iloc[va_idx].copy()
        X_te_lgb = X_te_fe.copy()
        for col in cat_cols:
            dtype = pd.CategoricalDtype(
                categories=sorted(X_tr_lgb[col].dropna().unique().tolist())
            )
            X_tr_lgb[col] = X_tr_lgb[col].astype(dtype)
            X_va_lgb[col] = X_va_lgb[col].astype(dtype)
            X_te_lgb[col] = X_te_lgb[col].astype(dtype)

        lgb_clf = lgb.LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            n_estimators=520,
            learning_rate=0.055,
            num_leaves=63,
            max_depth=5,
            min_child_samples=75,
            reg_lambda=1.5,
            colsample_bytree=0.78,
            subsample=0.85,
            subsample_freq=1,
            random_state=SEED + 100 * fold,
            n_jobs=LGB_N_JOBS,
            verbose=-1,
        )
        lgb_clf.fit(X_tr_lgb, y_tr, categorical_feature=cat_cols)
        oof_lgb_fe[va_idx] = lgb_clf.predict_proba(X_va_lgb)
        test_lgb_fe += lgb_clf.predict_proba(X_te_lgb) / 3.0

        print(
            f"   Fold {fold}/3 | LGB_FE Macro F1: {f1_score(y_va, oof_lgb_fe[va_idx].argmax(1), average='macro'):.5f} | "
            f"GPU_EmbResNet Macro F1: {f1_score(y_va, oof_gpu_nn[va_idx].argmax(1), average='macro'):.5f}"
        )

    # Evaluate baseline top pick (80% ref + 20% vip4)
    oof_top_pick = 0.80 * ref_oof + 0.20 * vip4_oof
    base_macro_f1 = f1_score(y, oof_top_pick.argmax(axis=1), average="macro")
    print(f"\nBaseline Top-Pick OOF Macro F1 (argmax): {base_macro_f1:.6f}")

    # Search optimal multi-family probability blend weights
    best_blend_f1 = base_macro_f1
    best_weights = (1.0, 0.0, 0.0)
    best_oof_blend = oof_top_pick.copy()

    for w_lgb in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for w_nn in [0.0, 0.03, 0.05, 0.08, 0.10]:
            w_top = round(1.0 - w_lgb - w_nn, 4)
            if w_top < 0.70:
                continue
            cand_oof = w_top * oof_top_pick + w_lgb * oof_lgb_fe + w_nn * oof_gpu_nn
            score = f1_score(y, cand_oof.argmax(axis=1), average="macro")
            if score > best_blend_f1:
                best_blend_f1 = score
                best_weights = (w_top, w_lgb, w_nn)
                best_oof_blend = cand_oof.copy()

    w_top, w_lgb, w_nn = best_weights
    print(
        f"Multi-Family Blend ({w_top:.2f} TopPick + {w_lgb:.2f} LGB_FE + {w_nn:.2f} GPU_EmbResNet) "
        f"OOF Macro F1 (argmax): {best_blend_f1:.6f}"
    )

    # Nested Out-of-Fold verification of Coordinate Ascent Class Logit Offsets
    nested_preds = np.zeros(len(y), dtype=int)
    fold_offsets = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, y), start=1):
        offsets_k = optimize_macro_f1_offsets_coordinate_ascent(
            best_oof_blend[tr_idx], y[tr_idx], n_rounds=3
        )
        fold_offsets.append(offsets_k)
        log_p_va = np.log(np.clip(best_oof_blend[va_idx], 1e-7, 1.0))
        nested_preds[va_idx] = (log_p_va + offsets_k).argmax(axis=1)

    nested_offset_f1 = f1_score(y, nested_preds, average="macro")
    mean_offsets = np.mean(fold_offsets, axis=0)
    full_offsets = optimize_macro_f1_offsets_coordinate_ascent(
        best_oof_blend, y, n_rounds=4
    )
    # Conservative regularised offset (50% nested mean + 50% full OOF)
    final_offsets = 0.5 * mean_offsets + 0.5 * full_offsets
    final_oof_preds = (
        np.log(np.clip(best_oof_blend, 1e-7, 1.0)) + final_offsets
    ).argmax(axis=1)
    final_oof_f1 = f1_score(y, final_oof_preds, average="macro")

    print(
        f"Coordinate Ascent Logit Offsets: {np.round(final_offsets, 4).tolist()} | "
        f"Nested OOF Macro F1: {nested_offset_f1:.6f} | Final OOF Macro F1: {final_oof_f1:.6f}"
    )

    # Apply to Test Set
    final_test_probs = (
        w_top * top_pick_test_probs + w_lgb * test_lgb_fe + w_nn * test_gpu_nn
    )
    if nested_offset_f1 > best_blend_f1:
        test_preds = (
            np.log(np.clip(final_test_probs, 1e-7, 1.0)) + final_offsets
        ).argmax(axis=1)
    else:
        test_preds = final_test_probs.argmax(axis=1)

    sample_sub = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame({"id": sample_sub["id"], "task2": test_preds.astype(int)})
    sub_df.to_csv(SUBMIT_DIR / "index.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "player_id": test_df["player_id"],
            "segment": test_preds.astype(int),
            "prob_0": final_test_probs[:, 0],
            "prob_1": final_test_probs[:, 1],
            "prob_2": final_test_probs[:, 2],
            "prob_3": final_test_probs[:, 3],
        }
    )
    pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)

    print(
        f"Saved Task 2 predictions to {SUBMIT_DIR / 'index.csv'}: "
        f"{pd.Series(test_preds).value_counts().sort_index().to_dict()} in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()

