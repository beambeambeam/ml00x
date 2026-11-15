"""Task 5: unsupervised account anomaly detection.

Design idea ("each player has their own rectangle"):
    The user's lat/lon plot showed that every player's four logins sit in a small
    box around one home point, and players that leave their box are suspicious.
    We generalise that to every behaviour group:

    1. Personal baseline: periods 1-3 define what is normal for *this* player.
       Period 4 is compared against it (drift features, robust-scaled by the
       population's typical period-to-period noise).
    2. Population baseline: each player's 4-period average is compared against
       everyone else (level features), which catches always-abnormal accounts
       such as bots that never change.
    3. Geography: displacement of each login from the player's home point, plus
       coordinates outside the valid lat/lon box.

    Detectors on the engineered matrix:
      - HDBSCAN-CBLOF (primary): density clustering (HDBSCAN) per feature, then
        cluster-based outlier scoring: players in a small cluster (<5%) that an
        empty band separates from the large clusters are outliers. This is the
        "rectangle" test learned by a model. Full-space models (Isolation
        Forest, MCD, GMM, autoencoder, LOF) miss the bot, silent, low-activity
        and unstable clusters because each differs in only one or a few of ~75
        features; see assets/model-comparison.csv.
      - Robust tail score: soft-max of |robust z| over features.
      - Isolation Forest on the clipped robust-z matrix (multi-feature combos).
      - Gap box: a simpler sorted-gap version of the HDBSCAN test, kept only as
        an independent cross-check (not in the vote).
    Hard vote: anomalous if HDBSCAN-CBLOF flags the player OR the tail score
    passes its natural break. F3 weights recall 9x, so a union suits it.

Test rows are only used without labels (allowed for Task 5).
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import IsolationForest

SEED = 42
PERIODS = (1, 2, 3, 4)
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
STATIC = ["account_age_days", "behavior_stability"]
FLAGS = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
# Count-like groups have a natural long right tail; their level z is less trustworthy.
DISCRETE = {"device_count", "refund_count", "guild_changes"}


def period_groups(df: pd.DataFrame) -> list[str]:
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c[-2:] in {f"_{p}" for p in PERIODS}})


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad < 1e-9:
        mad = x.std() or 1.0
    return (x - med) / mad


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    for g in period_groups(df):
        if g in {"login_lat", "login_lon"}:
            continue
        X = df[[f"{g}_{p}" for p in PERIODS]]
        base = X.iloc[:, :3].median(axis=1)
        feats[f"{g}__level"] = X.mean(axis=1)
        # Personal drift: period 4 vs own baseline. Missing -> no evidence (0).
        feats[f"{g}__drift4"] = (X.iloc[:, 3] - base).fillna(0.0)
        feats[f"{g}__spread"] = X.std(axis=1)

    lat = df[[f"login_lat_{p}" for p in PERIODS]].to_numpy()
    lon = df[[f"login_lon_{p}" for p in PERIODS]].to_numpy()
    with warnings.catch_warnings():  # players with all-NaN coordinates
        warnings.simplefilter("ignore", RuntimeWarning)
        home_lat, home_lon = np.nanmedian(lat[:, :3], 1), np.nanmedian(lon[:, :3], 1)
    disp = np.hypot(lat - home_lat[:, None], lon - home_lon[:, None])
    feats["geo__disp4"] = pd.Series(np.nan_to_num(disp[:, 3], nan=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats["geo__disp_max"] = pd.Series(np.nan_to_num(np.nanmax(disp, 1), nan=0.0))
    outside = (np.abs(lat) > 90) | (np.abs(lon) > 180)
    feats["geo__outside"] = pd.Series(outside.sum(1).astype(float))

    for c in STATIC:
        feats[c] = df[c]
    F = pd.DataFrame(feats)
    return F.fillna(F.median())


def score(F: pd.DataFrame):
    Z = F.apply(robust_z)
    for c in Z:
        if c.split("__")[0] in DISCRETE:
            Z[c] = Z[c] * 0.5  # damp coarse integer features
    Zc = Z.clip(-30, 30)

    # Detector 1: soft-max of absolute robust z (smooth "any feature out of its box").
    A = Zc.abs().to_numpy()
    tail = np.log(np.exp(A / 3).sum(1)) * 3

    # Detector 2: Isolation Forest on the robust-z matrix.
    iso = IsolationForest(n_estimators=600, max_samples=1024, random_state=SEED, n_jobs=-1)
    iso.fit(Zc)
    iforest = -iso.score_samples(Zc)

    boxes = gap_boxes(F)
    box_out = pd.DataFrame({c: (F[c] < lo) | (F[c] > hi) for c, (lo, hi) in boxes.items()})
    pd.DataFrame([{"feature": c, "low": lo, "high": hi, "players_outside": int(box_out[c].sum())}
                  for c, (lo, hi) in boxes.items()]).to_csv(ASSETS / "gap-boxes.csv", index=False)

    outside, clusters = hdbscan_cblof(F)
    clusters.sort_values("separation", ascending=False).to_csv(ASSETS / "hdbscan-clusters.csv", index=False)

    out = pd.DataFrame({"tail": tail, "iforest": iforest,
                        "hdbscan_hits": outside.sum(axis=1).to_numpy(),
                        "box_hits": box_out.sum(axis=1).to_numpy()})
    ranks = out[["tail", "iforest"]].rank(pct=True)
    out["ensemble"] = ranks.mean(axis=1)  # soft vote on ranks
    return out, outside, boxes


def hdbscan_cblof(F: pd.DataFrame, small_frac=0.05, min_sep=100.0, eps=0.1, min_unique=200):
    """Per-feature HDBSCAN + cluster-based outlier factor.

    For each feature (scaled by IQR), HDBSCAN finds density clusters;
    `cluster_selection_epsilon` merges fragments of the same dense core. A
    cluster smaller than `small_frac` of players is an outlier cluster when the
    empty band between it and the nearest large cluster is at least `min_sep`
    times the core's typical spacing near that edge. Measured on this data, real
    clusters sit at >=125x and tail fragments at <=63x, so 100 falls in that gap.
    Returns per-feature membership flags and a per-cluster table.
    """
    hits, table = {}, []
    for c in F:
        v = F[c].to_numpy()
        if len(np.unique(v)) < min_unique:
            continue
        x = (v - np.median(v)) / (np.subtract(*np.percentile(v, [75, 25])) + 1e-9)
        lab = HDBSCAN(min_cluster_size=50, min_samples=20, cluster_selection_epsilon=eps,
                      copy=True).fit_predict(x.reshape(-1, 1))
        sizes = pd.Series(lab[lab >= 0]).value_counts()
        big = sizes[sizes >= small_frac * len(x)].index
        core = np.sort(x[np.isin(lab, big)])
        if len(core) == 0:
            continue
        flag = np.zeros(len(x), bool)
        for sc in sizes[sizes < small_frac * len(x)].index:
            m = lab == sc
            members = x[m]
            if len(np.unique(members)) < 0.5 * m.sum():
                continue  # point mass (e.g. never adds friends): legit sub-population
            low_side = members.mean() < np.median(core)
            gap = core[0] - members.max() if low_side else members.min() - core[-1]
            edge = core[:200] if low_side else core[-200:]
            spacing = np.median(np.diff(edge)) + 1e-12
            sep = gap / spacing
            table.append({"feature": c, "cluster_size": int(m.sum()), "gap_iqr": round(float(gap), 4),
                          "separation": round(float(sep), 1), "outlier_cluster": bool(sep >= min_sep)})
            if sep >= min_sep:
                flag |= m
        if flag.any():
            hits[c] = flag
    return pd.DataFrame(hits, index=F.index), pd.DataFrame(table)


def gap_boxes(F: pd.DataFrame, tail_frac=0.06, min_ratio=300.0, min_group=20, min_unique=200):
    """Learn a [low, high] box per feature from density gaps in its outer tails.

    A tail edge is used only when the widest spacing in that tail is
    `min_ratio` times larger than the typical spacing there, i.e. a real empty
    band, and at least `min_group` players sit beyond it.
    """
    boxes = {}
    for c in F:
        v = np.sort(F[c].to_numpy())
        if len(np.unique(v)) < min_unique:
            continue  # integer-like features always have gaps
        n, k = len(v), int(len(v) * tail_frac)
        lo, hi = -np.inf, np.inf
        for side, seg in (("lo", v[: k + 1]), ("hi", v[n - k - 1 :])):
            d = np.diff(seg)
            typical = np.median(d[d > 0]) if (d > 0).any() else 0.0
            if typical <= 0:
                continue
            i = int(np.argmax(d))
            beyond = i + 1 if side == "lo" else len(seg) - 1 - i
            group = seg[: i + 1] if side == "lo" else seg[i + 1 :]
            # A point mass (e.g. players who never add friends) is a legit
            # sub-population, not a separated anomaly cluster.
            point_mass = len(np.unique(group)) < 0.5 * len(group)
            if d[i] / typical >= min_ratio and beyond >= min_group and not point_mass:
                edge = (seg[i] + seg[i + 1]) / 2
                lo, hi = (edge, hi) if side == "lo" else (lo, edge)
        if np.isfinite(lo) or np.isfinite(hi):
            boxes[c] = (lo, hi)
    return boxes


def natural_break(s: np.ndarray, lo_q=0.85, hi_q=0.995) -> float:
    """Threshold at the widest gap in the upper tail (sorted score spacing)."""
    v = np.sort(s)
    lo, hi = int(len(v) * lo_q), int(len(v) * hi_q)
    seg = v[lo:hi]
    # Smooth spacing so a single pair doesn't dominate.
    gaps = np.convolve(np.diff(seg), np.ones(5) / 5, mode="same")
    i = int(np.argmax(gaps))
    return float(seg[i])


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "test.csv")
    F = engineer(df)
    S, outside, boxes = score(F)

    thr = natural_break(S["tail"].to_numpy())
    by_tail, by_hdb, by_box = S["tail"] > thr, S["hdbscan_hits"] > 0, S["box_hits"] > 0
    pred = (by_tail | by_hdb).astype(int)
    print(f"hdbscan: {by_hdb.sum()}  tail>{thr:.2f}: {by_tail.sum()}  both: {(by_tail & by_hdb).sum()}")
    print(f"cross-check gap box: {by_box.sum()} flagged, {(by_box & by_hdb).sum()} shared with hdbscan")
    print(f"flagged={pred.sum()} ({pred.mean():.2%})")

    pd.DataFrame({"player_id": df.player_id, **S, "is_anomaly": pred}).to_csv(
        ROOT / "predictions" / "detector-scores.csv", index=False)

    ids = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv", usecols=["id"])["id"]
    assert len(ids) == len(df), "row count mismatch with sample submission"
    (ROOT / "submit").mkdir(exist_ok=True)
    pd.DataFrame({"id": ids, "task5": pred.to_numpy()}).to_csv(ROOT / "submit" / "index.csv", index=False)

    plot_scores(S, thr)
    explain(df, F, outside, boxes, pred)


def plot_scores(S: pd.DataFrame, thr: float) -> None:
    ASSETS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(S["tail"], bins=200, color="#245a9b")
    axes[0].axvline(thr, color="#c0392b", ls="--", label=f"threshold {thr:.2f}")
    axes[0].set(yscale="log", xlabel="Robust tail score", ylabel="Players (log)", title="Tail score distribution")
    axes[0].legend()
    axes[1].scatter(S["iforest"], S["tail"], s=2, alpha=0.3, color="#245a9b", rasterized=True)
    axes[1].axhline(thr, color="#c0392b", ls="--")
    axes[1].set(xlabel="Isolation Forest score", ylabel="Robust tail score", title="Detector agreement")
    fig.savefig(ASSETS / "anomaly-score-distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


ARCHETYPES = [  # (name, box features); first match wins. Names are our reading of each cluster.
    ("vpn_pricing_abuse", ["purchase_frequency__drift4", "purchase_frequency__spread"]),
    ("account_takeover", ["currency_transfers__drift4", "currency_transfers__spread",
                          "play_pattern_consistency__drift4", "location_changes__drift4",
                          "geo__disp4", "geo__disp_max"]),
    ("account_sale_boosting", ["skill_rating__drift4", "skill_trajectory__drift4",
                               "skill_trajectory__spread", "friend_removes__drift4",
                               "friend_removes__spread"]),
    ("bot_24_7", ["login_time_variance__level", "ip_hash_entropy__level",
                  "session_duration_std__level", "kda_std__level"]),
    ("silent_account", ["communication_frequency__level"]),
    ("low_activity", ["activity_level__level"]),
    ("unstable_behavior", ["behavior_stability"]),
]


def archetypes(outside: pd.DataFrame, pred: pd.Series) -> pd.Series:
    label = pd.Series(np.where(pred.to_numpy() == 1, "other_tail_outlier", "normal"), index=outside.index)
    for name, cols in reversed(ARCHETYPES):
        hit = outside[[c for c in cols if c in outside]].any(axis=1)
        label[hit] = name
    return label


def explain(df: pd.DataFrame, F: pd.DataFrame, outside: pd.DataFrame, boxes: dict, pred: pd.Series) -> None:
    label = archetypes(outside, pred)
    counts = label[label != "normal"].value_counts().rename_axis("archetype").reset_index(name="players")
    counts["share_of_all_players"] = (counts["players"] / len(df)).round(4)
    counts.to_csv(ASSETS / "anomaly-archetypes.csv", index=False)
    print(counts.to_string(index=False))

    # One panel per box feature: the dense core vs the group beyond the gap.
    show = [c for c in ["geo__disp_max", "skill_trajectory__drift4", "currency_transfers__drift4",
                        "purchase_frequency__drift4", "ip_hash_entropy__level",
                        "communication_frequency__level", "activity_level__level",
                        "behavior_stability"] if c in boxes]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.flat, show):
        lo, hi = boxes[c]
        v = F[c]
        inside = (v >= lo) & (v <= hi)
        bins = np.linspace(v.min(), v.max(), 120)
        ax.hist(v[inside], bins=bins, color="#245a9b", label="inside box")
        ax.hist(v[~inside], bins=bins, color="#c0392b", label="outside box")
        for edge in (lo, hi):
            if np.isfinite(edge):
                ax.axvline(edge, color="#333", ls="--", lw=0.8)
        ax.set(yscale="log", title=c.replace("__", " · "), xlabel="value")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Gap boxes: the dense core of each feature vs the players beyond an empty gap",
                 x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "gap-boxes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"player_id": df.player_id, "archetype": label}).to_csv(
        ROOT / "predictions" / "archetypes.csv", index=False)


if __name__ == "__main__":
    main()
