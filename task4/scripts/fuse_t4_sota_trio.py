"""Pure-Train Dual-Regime SOTA Fusion & Shift-Gated Calibration for Task 4.

Auto-discovers completed full-resolution Nightmare-Hardened SGKF bundles
(`cnxv3`, `dinov3l`, `eva02l`, `cnxv2b`) alongside `task4-sgkf-e2e-bundle.npz`
and the 4 base ViT streams, then performs strictly zero-test-leakage Nested
5-Fold SGKF optimization:
1. Out-of-fold Temperature Scaling weighted by `train_hard_iso_w` (Guo et al., ICML 2017)
2. Clean/Standard Nested SLSQP Log-Odds Pooling (evicts low-res `CNXv2-StageC-2TTA` once full-res `cnxv3` exists)
3. Deterministic ExtHard6 Nested SLSQP Log-Odds Pooling across all 6-TTA hardened families
4. Pure-Train Pointwise Restoration Shift Gate tuned inside training folds (clean 99th-pct JSD + entropy reduction)
5. Nested Uniform-Prior Bayes Macro-F1 Debiasing (alpha*) & 500-resample paired bootstrap vs Run 22

Usage:
    python project/task4/scripts/fuse_t4_sota_trio.py --task4-dir project/task4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss


CLASS_MAP: Dict[str, int] = {
    "7-11k": 0,
    "7-11K": 0,
    "fiveN": 1,
    "freefried": 2,
    "pube": 3,
    "roblock": 4,
}
DEFAULT_FULLRES_KEYS = ("cnxv3", "dinov3l", "eva02l", "cnxv2b")
HARDENED_LABELS = {
    "cnxv3": "CNXv3-Base",
    "dinov3l": "DINOv3L-Hard",
    "eva02l": "EVA02L-Hard",
    "cnxv2b": "CNXv2B-Hard",
}


def norm_p(p: np.ndarray) -> np.ndarray:
    """Clip and row-normalize probability matrix to simplex."""
    arr = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0)
    return arr / arr.sum(axis=1, keepdims=True)


def entropy_norm(p: np.ndarray) -> np.ndarray:
    """Normalized predictive Shannon entropy H(p)/log(5) in [0, 1]."""
    arr = norm_p(p)
    return -np.sum(arr * np.log(arr), axis=1) / np.log(5.0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence D_JS(p || q)."""
    pa, qa = norm_p(p), norm_p(q)
    m = 0.5 * (pa + qa)
    return 0.5 * np.sum(pa * np.log(pa / m), axis=1) + 0.5 * np.sum(qa * np.log(qa / m), axis=1)


def geo_pool(models: List[np.ndarray], weights: np.ndarray, temps: List[float]) -> np.ndarray:
    """Temperature-scaled weighted log-odds geometric pooling."""
    lp = np.zeros_like(models[0], dtype=np.float64)
    for p, w, t in zip(models, weights, temps):
        if float(w) <= 0.0:
            continue
        lp += float(w) * (np.log(np.clip(p, 1e-7, 1.0)) / float(t))
    lp -= lp.max(axis=1, keepdims=True)
    ep = np.exp(lp)
    return ep / ep.sum(axis=1, keepdims=True)


def fit_nested_temps_weighted(
    models: List[np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    sample_w: np.ndarray,
) -> List[float]:
    """Fit per-stream out-of-fold temperatures strictly within train SGKF folds."""
    temps: List[float] = []
    for p in models:
        log_p = np.log(np.clip(p, 1e-7, 1.0))
        fold_ts: List[float] = []
        for k in range(5):
            tr = folds != k
            wt = sample_w[tr] / sample_w[tr].mean()
            tr_idx = np.arange(tr.sum())

            def _nll(inv_t: np.ndarray) -> float:
                sc = log_p[tr] * inv_t[0]
                sc -= sc.max(axis=1, keepdims=True)
                pr = np.exp(sc)
                pr /= pr.sum(axis=1, keepdims=True)
                return float(-np.mean(wt * np.log(np.clip(pr[tr_idx, y[tr]], 1e-12, 1.0))))

            res = minimize(_nll, [1.0], bounds=[(0.3, 2.0)], method="L-BFGS-B")
            fold_ts.append(float(1.0 / res.x[0]))
        temps.append(float(np.mean(fold_ts)))
    return temps


def fit_nested_pool_weights(
    models: List[np.ndarray],
    temps: List[float],
    y: np.ndarray,
    folds: np.ndarray,
    sample_w: np.ndarray,
    bounds: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Fit temperature-calibrated log-odds weights via Nested 5-Fold SGKF SLSQP."""
    fold_ws: List[np.ndarray] = []
    oof_out = np.zeros_like(models[0], dtype=np.float64)
    m = len(models)
    for k in range(5):
        tr, va = folds != k, folds == k
        wt = sample_w[tr] / sample_w[tr].mean()
        tr_idx = np.arange(tr.sum())
        lps_tr = [np.log(np.clip(p[tr], 1e-7, 1.0)) / t for p, t in zip(models, temps)]

        def _obj(w: np.ndarray) -> float:
            w = np.maximum(w, 0.0)
            w = w / (w.sum() + 1e-12)
            lp = sum(wi * lp_i for wi, lp_i in zip(w, lps_tr))
            lp -= lp.max(axis=1, keepdims=True)
            ep = np.exp(lp)
            pr = ep / ep.sum(axis=1, keepdims=True)
            return float(-np.mean(wt * np.log(np.clip(pr[tr_idx, y[tr]], 1e-12, 1.0))))

        res = minimize(_obj, np.ones(m) / m, bounds=bounds, method="SLSQP")
        wk = np.maximum(res.x, 0.0)
        wk /= wk.sum()
        fold_ws.append(wk)
        oof_out[va] = geo_pool([p[va] for p in models], wk, temps)
    return np.mean(fold_ws, axis=0), oof_out, fold_ws


def apply_shift_gate_blend(
    p_clean: np.ndarray,
    p_hard: np.ndarray,
    p_orig2_ref: np.ndarray,
    p_hard6_ref: np.ndarray,
    tau_js: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend log-odds toward ExtHard stack strictly where 6-TTA restoration shifts & reduces entropy."""
    if gamma <= 0.0:
        return p_clean.copy(), np.zeros(len(p_clean), dtype=np.float64)
    js = js_divergence(p_orig2_ref, p_hard6_ref)
    h_clean = entropy_norm(p_clean)
    h_hard = entropy_norm(p_hard)
    excess = np.clip((js - tau_js) / max(tau_js, 1e-4), 0.0, 1.0)
    gate = np.where((js > tau_js) & (h_hard < h_clean), gamma * excess, 0.0)
    lp = (1.0 - gate[:, None]) * np.log(norm_p(p_clean)) + gate[:, None] * np.log(norm_p(p_hard))
    lp -= lp.max(axis=1, keepdims=True)
    ep = np.exp(lp)
    return ep / ep.sum(axis=1, keepdims=True), gate


def fit_nested_shift_gate(
    p_clean_oof: np.ndarray,
    p_hard_oof: np.ndarray,
    p_orig2_oof: np.ndarray,
    p_hard6_oof: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    iso_mask: np.ndarray,
    sample_w: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Tune Pure-Train restoration shift gate strictly within outer training folds."""
    js_all = js_divergence(p_orig2_oof, p_hard6_oof)
    gammas = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35]
    fold_taus: List[float] = []
    fold_gammas: List[float] = []
    gated_oof = np.zeros_like(p_clean_oof, dtype=np.float64)

    for k in range(5):
        tr, va = folds != k, folds == k
        tau_k = float(np.quantile(js_all[tr & iso_mask], 0.99))
        wt = sample_w[tr] / sample_w[tr].mean()
        tr_idx = np.arange(tr.sum())
        best_g, best_nll = 0.0, float("inf")
        for g in gammas:
            pr_tr, _ = apply_shift_gate_blend(
                p_clean_oof[tr], p_hard_oof[tr], p_orig2_oof[tr], p_hard6_oof[tr], tau_k, g
            )
            nll = float(-np.mean(wt * np.log(np.clip(pr_tr[tr_idx, y[tr]], 1e-12, 1.0))))
            if nll < best_nll - 1e-9:
                best_nll, best_g = nll, g
        fold_taus.append(tau_k)
        fold_gammas.append(best_g)
        gated_oof[va], _ = apply_shift_gate_blend(
            p_clean_oof[va], p_hard_oof[va], p_orig2_oof[va], p_hard6_oof[va], tau_k, best_g
        )

    return gated_oof, float(np.mean(fold_taus)), float(np.mean(fold_gammas))


def fit_nested_prior_debias(
    oof_raw: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> Tuple[float, List[float], np.ndarray]:
    """Fit Pure-Train Uniform-Prior Bayes Macro-F1 multiplier alpha* via Nested 5-Fold SGKF."""
    alpha_grid = list(np.linspace(0.0, 1.0, 21))
    nested_deb = np.zeros_like(oof_raw, dtype=np.float64)
    fold_alphas: List[float] = []
    for k in range(5):
        tr, va = folds != k, folds == k
        pi_k = np.bincount(y[tr], minlength=5).astype(np.float64) / tr.sum()
        sw_k = 1.0 / pi_k[y[tr]]
        best_a, best_f1 = 0.0, -1.0
        for a in alpha_grid:
            pa = oof_raw[tr] / (pi_k[None, :] ** a)
            f1_k = float(f1_score(y[tr], pa.argmax(1), average="macro", sample_weight=sw_k))
            if f1_k > best_f1:
                best_f1, best_a = f1_k, float(a)
        fold_alphas.append(best_a)
        pv = oof_raw[va] / (pi_k[None, :] ** best_a)
        nested_deb[va] = pv / pv.sum(axis=1, keepdims=True)
    return float(np.mean(fold_alphas)), fold_alphas, nested_deb


def run_fusion(task4_dir: Path, save_outputs: bool = True, seed: int = 42) -> Dict[str, object]:
    """Run full Dual-Regime Pure-Train SOTA fusion and return verified metrics + artifact paths."""
    data_dir = task4_dir / "data"
    ml_dir = task4_dir / "predictions" / "model-level"
    pred_dir = task4_dir / "predictions"
    submit_dir = task4_dir / "submit"
    models_dir = task4_dir / "models"
    for d in (pred_dir, submit_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    y = (
        train_df["label"].astype(int).to_numpy()
        if "label" in train_df.columns
        else train_df["class"].map(CLASS_MAP).astype(int).to_numpy()
    )

    e2e = np.load(ml_dir / "task4-sgkf-e2e-bundle.npz")
    folds = e2e["folds"].astype(int)
    w_hard_iso = e2e["train_hard_iso_w"].astype(np.float64)
    iso_mask = e2e["train_iso_mask"].astype(bool)
    corr_mask = e2e["train_corr_mask"].astype(bool)

    p_cols = [f"prob_{i}" for i in range(5)]
    base_vit_names = ["dinov3-vitl", "eva02-large", "eva02-base", "dinov3-vitb"]
    vit_oof: Dict[str, np.ndarray] = {}
    vit_te: Dict[str, np.ndarray] = {}
    ref_test_ids = None
    ref_test_files = None
    for m in base_vit_names:
        o_df = pd.read_csv(ml_dir / f"task4-{m}-oof.csv")
        t_df = pd.read_csv(ml_dir / f"task4-{m}-test-predictions.csv")
        if "file_name" in train_df.columns and "file_name" in o_df.columns:
            assert (o_df["file_name"].astype(str).values == train_df["file_name"].astype(str).values).all()
        elif "file_name" in o_df.columns:
            assert (o_df["file_name"].astype(str).values == (train_df["id"].astype(str) + ".jpg").values).all()
        else:
            assert (o_df["id"].astype(str).values == train_df["id"].astype(str).values).all()
        if "file_name" in test_df.columns:
            assert (t_df["id"].astype(str).values == test_df["id"].astype(str).values).all()
            assert (t_df["file_name"].astype(str).values == test_df["file_name"].astype(str).values).all()
        else:
            assert set(t_df["file_name"].astype(str).values) == set((test_df["id"].astype(str) + ".jpg").values)
        if ref_test_ids is None:
            ref_test_ids = t_df["id"].astype(str).values
            ref_test_files = t_df["file_name"].astype(str).values if "file_name" in t_df.columns else None
        else:
            assert (t_df["id"].astype(str).values == ref_test_ids).all()
            if ref_test_files is not None and "file_name" in t_df.columns:
                assert (t_df["file_name"].astype(str).values == ref_test_files).all()
        vit_oof[m] = norm_p(o_df[p_cols].to_numpy())
        vit_te[m] = norm_p(t_df[p_cols].to_numpy())

    # Discover all completed Full-Res Nightmare-Hardened SGKF bundles
    discovered_fullres: Dict[str, np.lib.npyio.NpzFile] = {}
    for key in DEFAULT_FULLRES_KEYS:
        bpath = ml_dir / f"task4-{key}-sgkf-hard-bundle.npz"
        if bpath.is_file():
            discovered_fullres[key] = np.load(bpath)

    # Assemble Run 22 reference (9 streams) for paired bootstrap comparison
    r22_oof = [
        norm_p(e2e["cnx_a_oof"]),
        norm_p(e2e["din_a_oof"]),
        norm_p(e2e["mlp_oof"]),
        vit_oof["dinov3-vitl"],
        vit_oof["eva02-large"],
        vit_oof["eva02-base"],
        vit_oof["dinov3-vitb"],
        norm_p(e2e["din_c2_oof"]),
        norm_p(e2e["din_c6_oof"]),
    ]
    T_r22 = fit_nested_temps_weighted(r22_oof, y, folds, w_hard_iso)
    _, r22_raw, _ = fit_nested_pool_weights(r22_oof, T_r22, y, folds, w_hard_iso, [(0.0, 0.60)] * len(r22_oof))
    _, _, r22_deb = fit_nested_prior_debias(r22_raw, y, folds)

    # Primary Clean/Standard pool: Run 22 streams + completed Full-Res Hardened Orig2 & Hard6 streams
    # Note: Low-res CNXv2-StageC-2TTA (128x256) is excluded once full-res cnxv3 (192x384) is present.
    pool_names = [
        "ConvNeXtV2-SGKF",
        "DINOv2-StageA",
        "MLP-V3-SGKF",
        "DINOv3-ViT-L",
        "EVA02-Large",
        "EVA02-Base",
        "DINOv3-ViT-B",
        "DINOv2-StageC-2TTA",
        "DINOv2-StageC-6TTA",
    ]
    pool_oof = list(r22_oof)
    pool_te = [
        norm_p(e2e["cnx_a_te"]),
        norm_p(e2e["din_a_te"]),
        norm_p(e2e["mlp_te"]),
        vit_te["dinov3-vitl"],
        vit_te["eva02-large"],
        vit_te["eva02-base"],
        vit_te["dinov3-vitb"],
        norm_p(e2e["din_c2_te"]),
        norm_p(e2e["din_c6_te"]),
    ]

    if "cnxv3" not in discovered_fullres:
        pool_names += ["CNXv2-StageC-2TTA", "CNXv2-StageC-6TTA"]
        pool_oof += [norm_p(e2e["cnx_c2_oof"]), norm_p(e2e["cnx_c6_oof"])]
        pool_te += [norm_p(e2e["cnx_c2_te"]), norm_p(e2e["cnx_c6_te"])]

    for key, bun in discovered_fullres.items():
        lbl = HARDENED_LABELS.get(key, key)
        pool_names += [f"{lbl}-Orig2", f"{lbl}-Hard6"]
        pool_oof += [norm_p(bun["oof_orig2"]), norm_p(bun["oof_w6"])]
        pool_te += [norm_p(bun["te_orig2"]), norm_p(bun["te_w6"])]

    T_pool = fit_nested_temps_weighted(pool_oof, y, folds, w_hard_iso)
    w_clean, clean_oof_raw, _ = fit_nested_pool_weights(
        pool_oof, T_pool, y, folds, w_hard_iso, [(0.0, 0.60)] * len(pool_oof)
    )
    clean_te_raw = geo_pool(pool_te, w_clean, T_pool)

    # ExtHard6 Stress-Test stack across all available 6-TTA hardened families
    hard_names = ["CNXv2-StageC-6TTA", "DINOv2-StageC-6TTA"]
    hard_ext_oof = [norm_p(e2e["cnx_c6_ext"]), norm_p(e2e["din_c6_ext"])]
    hard_std_oof = [norm_p(e2e["cnx_c6_oof"]), norm_p(e2e["din_c6_oof"])]
    hard_te = [norm_p(e2e["cnx_c6_te"]), norm_p(e2e["din_c6_te"])]
    for key, bun in discovered_fullres.items():
        lbl = HARDENED_LABELS.get(key, key)
        hard_names.append(f"{lbl}-Hard6")
        hard_ext_oof.append(norm_p(bun["ext_w6"]))
        hard_std_oof.append(norm_p(bun["oof_w6"]))
        hard_te.append(norm_p(bun["te_w6"]))

    T_hard = [T_pool[pool_names.index(n)] if n in pool_names else 1.0 for n in hard_names]
    if "CNXv2-StageC-6TTA" not in pool_names:
        T_hard[0] = fit_nested_temps_weighted([hard_std_oof[0]], y, folds, w_hard_iso)[0]

    w_hard, ext_hard_oof, hard_fold_ws = fit_nested_pool_weights(
        hard_ext_oof, T_hard, y, folds, np.ones(len(y), dtype=np.float64), [(0.05, 0.80)] * len(hard_names)
    )
    hard_std_oof_pool = np.zeros_like(clean_oof_raw)
    for k in range(5):
        va = folds == k
        hard_std_oof_pool[va] = geo_pool([p[va] for p in hard_std_oof], hard_fold_ws[k], T_hard)
    hard_te_pool = geo_pool(hard_te, w_hard, T_hard)

    # Reference pair for restoration divergence: primary un-hardened Orig2 vs Hard6 if present
    if "cnxv3" in discovered_fullres:
        ref_o2_oof = norm_p(discovered_fullres["cnxv3"]["oof_orig2"])
        ref_w6_oof = norm_p(discovered_fullres["cnxv3"]["oof_w6"])
        ref_o2_te = norm_p(discovered_fullres["cnxv3"]["te_orig2"])
        ref_w6_te = norm_p(discovered_fullres["cnxv3"]["te_w6"])
    else:
        ref_o2_oof, ref_w6_oof = norm_p(e2e["cnx_c2_oof"]), norm_p(e2e["cnx_c6_oof"])
        ref_o2_te, ref_w6_te = norm_p(e2e["cnx_c2_te"]), norm_p(e2e["cnx_c6_te"])

    gated_oof_raw, tau_js, opt_gamma = fit_nested_shift_gate(
        clean_oof_raw, hard_std_oof_pool, ref_o2_oof, ref_w6_oof, y, folds, iso_mask, w_hard_iso
    )
    gated_te_raw, test_gate_vals = apply_shift_gate_blend(
        clean_te_raw, hard_te_pool, ref_o2_te, ref_w6_te, tau_js, opt_gamma
    )

    opt_alpha, fold_alphas, final_oof_deb = fit_nested_prior_debias(gated_oof_raw, y, folds)
    pi_tr = np.bincount(y, minlength=5).astype(np.float64) / len(y)
    final_te_probs = gated_te_raw / (pi_tr[None, :] ** opt_alpha)
    final_te_probs /= final_te_probs.sum(axis=1, keepdims=True)

    # Metrics + 500-resample paired bootstrap vs Run 22
    all_f1 = float(f1_score(y, final_oof_deb.argmax(1), average="macro"))
    iso_f1 = float(f1_score(y[iso_mask], final_oof_deb[iso_mask].argmax(1), average="macro"))
    corr_f1 = float(f1_score(y[corr_mask], final_oof_deb[corr_mask].argmax(1), average="macro"))
    ext_f1 = float(f1_score(y, ext_hard_oof.argmax(1), average="macro"))
    oof_ll = float(log_loss(y, gated_oof_raw))
    ext_ll = float(log_loss(y, ext_hard_oof))

    rng = np.random.default_rng(seed)
    p_new_c, p_r22_c = final_oof_deb.argmax(1), r22_deb.argmax(1)
    n = len(y)
    boot_wins = sum(
        1
        for _ in range(500)
        if (
            lambda idx: f1_score(y[idx], p_new_c[idx], average="macro")
            > f1_score(y[idx], p_r22_c[idx], average="macro")
        )(rng.integers(0, n, size=n))
    )
    boot_rate = boot_wins / 500.0

    pred_labels = final_te_probs.argmax(1).astype(int)
    test_dist = {int(k): int(v) for k, v in pd.Series(pred_labels).value_counts().sort_index().items()}

    summary: Dict[str, object] = {
        "discovered_fullres_models": list(discovered_fullres.keys()),
        "clean_pool_weights": {n: round(float(w), 4) for n, w in zip(pool_names, w_clean) if w > 1e-4},
        "clean_pool_temps": {n: round(float(t), 4) for n, t in zip(pool_names, T_pool)},
        "exthard_weights": {n: round(float(w), 4) for n, w in zip(hard_names, w_hard)},
        "shift_gate": {
            "tau_js_99pct_iso": round(tau_js, 6),
            "gamma": round(opt_gamma, 4),
            "test_rows_gated": int((test_gate_vals > 0).sum()),
        },
        "prior_debias_alpha": round(opt_alpha, 4),
        "fold_alphas": [round(float(a), 3) for a in fold_alphas],
        "metrics": {
            "all_oof_f1": round(all_f1, 6),
            "iso_oof_f1": round(iso_f1, 6),
            "corrupt_oof_f1": round(corr_f1, 6),
            "exthard6_oof_f1": round(ext_f1, 6),
            "oof_logloss": round(oof_ll, 6),
            "exthard6_logloss": round(ext_ll, 6),
            "run22_baseline_f1": round(float(f1_score(y, r22_deb.argmax(1), average="macro")), 6),
            "paired_bootstrap_vs_r22": round(boot_rate, 4),
        },
        "test_distribution": test_dist,
    }

    if save_outputs:
        out_full = pd.DataFrame(
            {
                "id": ref_test_ids if ref_test_ids is not None else test_df["id"].values,
                "prob_class_0": np.round(final_te_probs[:, 0], 6),
                "prob_class_1": np.round(final_te_probs[:, 1], 6),
                "prob_class_2": np.round(final_te_probs[:, 2], 6),
                "prob_class_3": np.round(final_te_probs[:, 3], 6),
                "prob_class_4": np.round(final_te_probs[:, 4], 6),
                "predicted_class": pred_labels,
            }
        )
        out_full.to_csv(pred_dir / "test_predictions.csv", index=False)

        submit_df = pd.DataFrame({"id": [f"ANS{i:05d}" for i in range(1, len(test_df) + 1)], "task4": pred_labels})
        submit_df.to_csv(submit_dir / "index.csv", index=False)

        with open(models_dir / "t4_sota_fusion_meta.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Pure-Train Dual-Regime SOTA Fusion for Task 4")
    ap.add_argument(
        "--task4-dir",
        default="/Users/wisitsuwannao/Documents/Workspace/cpe342-machine-learning-2026/project/task4",
        help="Path to project/task4 directory",
    )
    ap.add_argument("--dry-run", action="store_true", help="Compute metrics without writing prediction/submit CSVs")
    args = ap.parse_args()

    res = run_fusion(Path(args.task4_dir), save_outputs=not args.dry_run)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
