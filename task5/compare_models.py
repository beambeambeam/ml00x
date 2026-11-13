"""Compare unsupervised detectors on the engineered features from main.py.

There are no labels, so this is not an accuracy table. Each model flags its
top-k players (k = size of the final flagged set) and we report how much of
each archetype it recovers. Archetypes come from predictions/archetypes.csv
(run main.py first). The point: full-space models miss archetypes that differ
in only one or a few features, which is why main.py uses per-feature
HDBSCAN-CBLOF as its primary detector.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import main as M

warnings.filterwarnings("ignore")


def main() -> None:
    df = pd.read_csv(M.ROOT / "data" / "test.csv")
    F = M.engineer(df)
    arch = pd.read_csv(M.ROOT / "predictions" / "archetypes.csv")["archetype"]
    scores = pd.read_csv(M.ROOT / "predictions" / "detector-scores.csv")
    flagged = (arch != "normal").to_numpy()
    k = int(flagged.sum())

    Z = F.apply(M.robust_z).clip(-30, 30)
    Q = QuantileTransformer(output_distribution="normal", n_quantiles=2000,
                            random_state=M.SEED).fit_transform(F)
    Xs = StandardScaler().fit_transform(Z)

    models = {
        "Isolation Forest (full space)": lambda: -IsolationForest(
            n_estimators=500, random_state=M.SEED, n_jobs=-1).fit(Z).score_samples(Z),
        "Robust Mahalanobis (MCD)": lambda: MinCovDet(
            random_state=M.SEED, support_fraction=0.9).fit(Q).mahalanobis(Q),
        "Gaussian Mixture (8 comp.)": lambda: -GaussianMixture(
            8, covariance_type="diag", random_state=M.SEED).fit(Z).score_samples(Z),
        "Local Outlier Factor": lambda: -LocalOutlierFactor(
            n_neighbors=50, n_jobs=-1).fit(Q).negative_outlier_factor_,
        "Autoencoder (64-16-64)": lambda: ((MLPRegressor(
            hidden_layer_sizes=(64, 16, 64), max_iter=60, random_state=M.SEED,
            early_stopping=True).fit(Xs, Xs).predict(Xs) - Xs) ** 2).mean(1),
        "Subspace Isolation Forest (per feature)": lambda: pd.DataFrame({
            c: -IsolationForest(n_estimators=200, max_samples=256, random_state=M.SEED)
            .fit(F[[c]]).score_samples(F[[c]])
            for c in F if F[c].nunique() >= 200}).max(axis=1).to_numpy(),
        "Robust tail score": lambda: scores["tail"].to_numpy(),
        "HDBSCAN-CBLOF (per feature)": lambda: scores["hdbscan_hits"].to_numpy()
        + 1e-6 * scores["tail"].rank().to_numpy(),
    }
    names = [a for a in arch.unique() if a not in ("normal", "other_tail_outlier")]
    rows = []
    for name, fn in models.items():
        s = np.asarray(fn(), dtype=float)
        top = s >= np.sort(s)[-k]
        row = {"model": name, "overlap_with_final_set": round((top & flagged).sum() / k, 3)}
        row.update({a: round(top[(arch == a).to_numpy()].mean(), 2) for a in names})
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(M.ASSETS / "model-comparison.csv", index=False)


if __name__ == "__main__":
    main()
