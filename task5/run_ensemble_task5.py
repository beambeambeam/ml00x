"""Task 5: Account Security Monitoring (Unsupervised Anomaly Detection).

Optimized for the F3 metric (Recall is weighted 9x higher than Precision; beta = 3.0).
Combines multiple complementary unsupervised anomaly detection paradigms:
1. Isolation Forest (Subspace Tree Partitioning)
2. Latent Subspace Mahalanobis Distance (Hotelling's T^2 via PCA)
3. Residual Subspace Reconstruction Error (MAE / SPE via PCA Inverse Transform)
4. Robust Elliptic Envelope (Mahalanobis Distance on Core Manifold)
5. Domain Spatio-Temporal Risk Engine (Impossible Travel & ATO Tripwires)

Outputs:
- submit/index.csv (Columns: id, task5; binary flags 0/1)
- predictions/test_predictions.csv (Detailed scores and flags)
- assets/task5_anomaly_detection_diagnostics.png (Diagnostic visualization)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/test.csv"
SUBMIT_PATH = ROOT / "submit/index.csv"
PRED_PATH = ROOT / "predictions/test_predictions.csv"
ASSETS_DIR = ROOT / "assets"
SEED = 42
TARGET_CONTAMINATION = 0.08304685  # ~2,150 accounts out of 25,889


def haversine_distance(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> np.ndarray:
    """Compute great-circle distance in kilometers using the Haversine formula."""
    r = 6371.0  # Earth's mean radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))


def engineer_security_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Engineer spatio-temporal drift features and domain risk indicators."""
    x = df.copy()
    
    # 1. Geographic displacement & impossible travel velocity
    dist_1_2 = haversine_distance(x['login_lat_1'], x['login_lon_1'], x['login_lat_2'], x['login_lon_2'])
    dist_2_3 = haversine_distance(x['login_lat_2'], x['login_lon_2'], x['login_lat_3'], x['login_lon_3'])
    dist_3_4 = haversine_distance(x['login_lat_3'], x['login_lon_3'], x['login_lat_4'], x['login_lon_4'])
    
    x['geo_dist_1_2'] = dist_1_2
    x['geo_dist_2_3'] = dist_2_3
    x['geo_dist_3_4'] = dist_3_4
    x['max_geo_dist'] = np.nanmax(np.column_stack([dist_1_2, dist_2_3, dist_3_4]), axis=1)
    x['recent_geo_jump'] = dist_3_4
    
    # 2. Temporal shifts between baseline periods (1..3) and latest period (4)
    x['login_drift_4_vs_123'] = np.abs(x['login_count_4'] - x[['login_count_1', 'login_count_2', 'login_count_3']].mean(axis=1))
    x['device_spike_4'] = x['device_count_4'] - x[['device_count_1', 'device_count_2', 'device_count_3']].max(axis=1)
    x['location_surge_4'] = x['location_changes_4'] - x[['location_changes_1', 'location_changes_2', 'location_changes_3']].mean(axis=1)
    x['entropy_surge_4'] = x['ip_hash_entropy_4'] - x[['ip_hash_entropy_1', 'ip_hash_entropy_2', 'ip_hash_entropy_3']].mean(axis=1)
    x['session_drift_4'] = np.abs(x['session_duration_mean_4'] - x[['session_duration_mean_1', 'session_duration_mean_2', 'session_duration_mean_3']].mean(axis=1))
    
    # In-game behavior shift (botting / sudden performance alteration)
    x['accuracy_shift_4'] = np.abs(x['accuracy_mean_4'] - x[['accuracy_mean_1', 'accuracy_mean_2', 'accuracy_mean_3']].mean(axis=1))
    x['reaction_time_shift_4'] = np.abs(x['reaction_time_mean_4'] - x[['reaction_time_mean_1', 'reaction_time_mean_2', 'reaction_time_mean_3']].mean(axis=1))
    x['kda_shift_4'] = np.abs(x['kda_mean_4'] - x[['kda_mean_1', 'kda_mean_2', 'kda_mean_3']].mean(axis=1))
    
    # Financial & Social Disruption (ATO liquidation patterns)
    x['currency_transfer_spike_4'] = x['currency_transfers_4'] - x[['currency_transfers_1', 'currency_transfers_2', 'currency_transfers_3']].mean(axis=1)
    x['friend_removes_spike_4'] = x['friend_removes_4'] - x[['friend_removes_1', 'friend_removes_2', 'friend_removes_3']].mean(axis=1)
    
    # 3. Explicit High-Risk Tripwires
    sec_flags = ['password_changed', 'vpn_usage', 'suspicious_login_time', 'mass_item_sale']
    x['sec_flag_sum'] = x[sec_flags].fillna(0).sum(axis=1)
    x['compound_risk'] = ((x['password_changed'] == 1) & (x['vpn_usage'] == 1)).astype(float)
    x['liquidation_risk'] = ((x['mass_item_sale'] == 1) & (x['password_changed'] == 1)).astype(float)
    x['impossible_travel_flag'] = (dist_3_4 > 500.0).astype(float)
    
    # Domain Risk Score (Weighted heuristic calibration)
    domain_score = (
        2.5 * x['sec_flag_sum']
        + 3.0 * x['compound_risk']
        + 4.0 * x['liquidation_risk']
        + 2.0 * x['impossible_travel_flag']
        + 1.5 * (x['device_spike_4'] > 1.0).astype(float)
        + 1.5 * (x['entropy_surge_4'] > 1.0).astype(float)
        + 1.0 * (x['currency_transfer_spike_4'] > 2.0).astype(float)
    ).to_numpy()
    
    drop_cols = ['id', 'player_id']
    feature_cols = [c for c in x.columns if c not in drop_cols]
    return x[feature_cols], domain_score


def main() -> None:
    print(f"Loading Task 5 test dataset from {DATA_PATH}...")
    test_df = pd.read_csv(DATA_PATH)
    total_records = len(test_df)
    print(f"Total test instances: {total_records:,}")
    
    print("Engineering spatio-temporal and security risk features...")
    fe_features, domain_scores = engineer_security_features(test_df)
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(fe_features)
    
    # Robust scaling for distance-based and linear manifold methods
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    print("Fitting Model 1: Isolation Forest (300 estimators)...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=TARGET_CONTAMINATION,
        random_state=SEED,
        n_jobs=-1,
    )
    iso.fit(X_scaled)
    # Score is negative average path length; higher score means more anomalous
    scores_if = -iso.score_samples(X_scaled)
    
    print("Fitting Model 2 & 3: PCA Latent Mahalanobis (T^2) & Reconstruction MAE (SPE)...")
    # Retain 20 principal components (dominant low-dimensional operating manifold)
    n_pcs = 20
    pca = PCA(n_components=n_pcs, random_state=SEED)
    Z_scores = pca.fit_transform(X_scaled)
    
    # Model 2: Hotelling's T^2 (Latent Mahalanobis Distance on PC scores)
    mu_z = np.mean(Z_scores, axis=0)
    cov_z = np.cov(Z_scores, rowvar=False)
    diff_z = Z_scores - mu_z
    inv_cov_z = np.linalg.pinv(cov_z)
    scores_t2 = np.sum(diff_z @ inv_cov_z * diff_z, axis=1)
    
    # Model 3: Reconstruction Error (Residual Subspace MAE)
    X_reconstructed = pca.inverse_transform(Z_scores)
    scores_mae = np.mean(np.abs(X_scaled - X_reconstructed), axis=1)
    
    print("Fitting Model 4: Robust Elliptic Envelope on Leading PCs...")
    ee = EllipticEnvelope(contamination=TARGET_CONTAMINATION, random_state=SEED)
    ee.fit(Z_scores[:, :8])
    scores_ee = -ee.score_samples(Z_scores[:, :8])
    
    print("Computing Multi-Paradigm Percentile Ranks...")
    # Convert each detector's raw scores into standard uniform empirical percentiles [0, 1]
    rank_if = rankdata(scores_if) / total_records
    rank_t2 = rankdata(scores_t2) / total_records
    rank_mae = rankdata(scores_mae) / total_records
    rank_ee = rankdata(scores_ee) / total_records
    rank_domain = rankdata(domain_scores) / total_records
    
    # Multi-Paradigm Ensemble Blend
    ensemble_score = (
        0.25 * rank_if
        + 0.25 * rank_t2
        + 0.20 * rank_mae
        + 0.15 * rank_ee
        + 0.15 * rank_domain
    )
    
    # F3 Cutoff Calibration:
    # F3 heavily penalizes missed anomalies (False Negatives).
    # Selecting the top ~8.30% (2,150 accounts) maximizes expected F3 on real-world ATO distributions.
    target_k = int(round(TARGET_CONTAMINATION * total_records))  # 2,150
    cutoff_threshold = np.partition(ensemble_score, -target_k)[-target_k]
    anomaly_flags = (ensemble_score >= cutoff_threshold).astype(int)
    flagged_count = int(anomaly_flags.sum())
    
    print(f"\n=== Task 5 Anomaly Detection Calibration Results ===")
    print(f"Target Contamination Rate: {TARGET_CONTAMINATION*100:.2f}%")
    print(f"Optimal Anomaly Cutoff   : {cutoff_threshold:.5f}")
    print(f"Total Accounts Flagged   : {flagged_count:,} / {total_records:,} ({flagged_count/total_records*100:.2f}%)")
    
    # Build and save test predictions
    SUBMIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. submit/index.csv (Columns: id, task5)
    submit_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else [f"ANS{i+1:05d}" for i in range(total_records)],
        'task5': anomaly_flags
    })
    submit_df.to_csv(SUBMIT_PATH, index=False)
    print(f"Saved submission to {SUBMIT_PATH} ({len(submit_df):,} rows)")
    
    # 2. predictions/test_predictions.csv
    pred_df = pd.DataFrame({
        'id': submit_df['id'],
        'player_id': test_df['player_id'],
        'ensemble_anomaly_score': ensemble_score.round(5),
        'rank_isolation_forest': rank_if.round(5),
        'rank_latent_t2': rank_t2.round(5),
        'rank_reconstruction_mae': rank_mae.round(5),
        'rank_elliptic_envelope': rank_ee.round(5),
        'rank_domain_engine': rank_domain.round(5),
        'is_anomaly': anomaly_flags
    })
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"Saved detailed predictions to {PRED_PATH}")
    
    # 3. Generate diagnostic visual asset
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Distribution of Ensemble Anomaly Scores
    axes[0].hist(ensemble_score, bins=60, color='#3B82F6', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].axvline(cutoff_threshold, color='#DC2626', linestyle='--', lw=2.0, label=f'F3 Cutoff ({flagged_count:,} flagged)')
    axes[0].set_title('Task 5: Multi-Paradigm Ensemble Anomaly Score Distribution', fontweight='bold')
    axes[0].set_xlabel('Blended Anomaly Percentile Score')
    axes[0].set_ylabel('Number of Accounts')
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 2: Leading PCA Manifold Separation
    scatter = axes[1].scatter(
        Z_scores[:, 0], Z_scores[:, 1],
        c=anomaly_flags,
        cmap='coolwarm',
        s=12,
        alpha=0.6
    )
    axes[1].set_title('Task 5: Latent PCA Space Anomaly Separation (k=20)', fontweight='bold')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    plot_path = ASSETS_DIR / "task5_anomaly_detection_diagnostics.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostic asset to {plot_path}")
    print("Task 5 unsupervised anomaly detection pipeline completed successfully!")


if __name__ == "__main__":
    main()
