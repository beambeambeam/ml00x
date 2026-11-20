"""Task 5: rule-based anomaly detection from the injected anomaly archetypes.

Normal players follow one pattern: each feature group is a personal base value
plus small period-to-period noise (period correlation 0.75-0.99). Anomalies are
injected as archetypes. Each archetype draws a few features from a uniform box
that is far from the normal range. These boxes have empty gaps to the normal
range, so exact rules separate them. Each rule fires on any observed signal,
which makes it robust to the ~5% NaN cells.

One archetype (unstable behaviour) has a single box feature,
`behavior_stability`. When that cell is NaN, a LightGBM model scores the player
from its weaker side effects (purchase frequency, login count, account age).
The model trains on players whose `behavior_stability` is observed, where the
rule label is exact. Out-of-fold isotonic calibration turns its scores into
probabilities. F3 gains from a flag when P(anomaly) > F3 / 10, so the cut
maximises expected F3.

Task 5 has only test.csv. The rules and the model use it without labels, which
the competition allows for this task.

Usage (from task5/):
    uv run archetype_rules.py
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PERIODS = (1, 2, 3, 4)
SEEDS = range(5)

# Erratic-combat box: all three groups are i.i.d. uniform in these ranges in every period.
ERRATIC_BOX = {"kda_std": (0.7, 1.0), "accuracy_std": (9.0, 12.0), "performance_stability": (0.85, 0.95)}


def cols(group: str) -> list[str]:
    return [f"{group}_{p}" for p in PERIODS]


def count(df: pd.DataFrame, group: str, cond) -> pd.Series:
    return cond(df[cols(group)]).sum(axis=1)


def archetype_rules(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per archetype. Cut-offs sit in the empty gap between
    the normal range (min/max over normal players in comments) and the box."""
    r = {}
    # Account takeover / VPN pricing abuse: period-4 jumps. Normal location_changes_4 <= 3.6,
    # purchase_frequency_4 <= 6.0, currency_transfers_4 <= 12.0, play_pattern_consistency_4 >= 0.55.
    r["account_takeover"] = (
        (df.location_changes_4 > 4.5)
        | (df.currency_transfers_4 > 13)
        | (df.play_pattern_consistency_4 < 0.5)
    )
    r["vpn_pricing_abuse"] = df.purchase_frequency_4 > 7.5
    # Account sale / boosting: normal skill_trajectory_4 <= 22.4, friend_removes_4 <= 4.8,
    # skill_rating_4 <= 2644.
    r["account_sale_boosting"] = (
        (df.skill_trajectory_4 > 30) | (df.friend_removes_4 > 6) | (df.skill_rating_4 > 2750)
    )
    # 24/7 bot: normal ip_hash_entropy >= 0.30, login_time_variance >= 1.4,
    # session_duration_std >= 6.9; bot boxes are [0, 0.2], [0.1, 0.5], [1, 5].
    bot_hits = (
        count(df, "ip_hash_entropy", lambda x: x < 0.25)
        + count(df, "login_time_variance", lambda x: x < 0.8)
        + count(df, "session_duration_std", lambda x: x < 6)
    )
    r["bot_24_7"] = bot_hits >= 2
    # Low activity: normal activity_level >= 0.35, box [0.1, 0.3].
    r["low_activity"] = count(df, "activity_level", lambda x: x < 0.33) >= 1
    # Silent account: normal communication_frequency >= 0.24, box [0, 0.1].
    r["silent_account"] = count(df, "communication_frequency", lambda x: x < 0.2) >= 1
    # Unstable behaviour: normal behavior_stability >= 0.70, box [0.2, 0.4].
    r["unstable_behavior"] = df.behavior_stability < 0.55
    # Erratic combat: every observed period of all three groups inside the box.
    inside = sum(
        ((df[c] >= lo) & (df[c] <= hi)).astype(int) for g, (lo, hi) in ERRATIC_BOX.items() for c in cols(g)
    )
    observed = sum(df[c].notna().astype(int) for g in ERRATIC_BOX for c in cols(g))
    r["erratic_combat"] = (observed >= 6) & (inside == observed)
    return pd.DataFrame(r)


def unstable_side_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features for unstable players, without behavior_stability.
    Hard filter: all 246 observed members have every purchase_frequency in [3, 6]
    and account_age_days >= 800."""
    pf, lc = df[cols("purchase_frequency")], df[cols("login_count")]
    hard = (((pf >= 3) & (pf <= 6)) | pf.isna()).all(axis=1) & (
        df.account_age_days.isna() | (df.account_age_days >= 795)
    )
    X = pd.concat([pf, lc], axis=1).assign(
        age=df.account_age_days,
        lc_mean=lc.mean(axis=1),
        lc_min=lc.min(axis=1),
        lc_max=lc.max(axis=1),
        pf_mean=pf.mean(axis=1),
        pf_sd=pf.std(axis=1),
        pf_n=pf.notna().sum(axis=1),
    )
    return X, hard


def unstable_nan_probability(df: pd.DataFrame, X: pd.DataFrame, hard: pd.Series):
    """Calibrated P(unstable) for players with NaN behavior_stability."""
    known = df.behavior_stability.notna()
    train = hard & known
    Xk, yk = X[train], (df.behavior_stability[train] < 0.55).astype(int).to_numpy()
    oof, models = np.zeros(len(Xk)), []
    for seed in SEEDS:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xk, yk):
            m = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=30,
                subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1,
            ).fit(Xk.iloc[tr], yk[tr])
            oof[va] += m.predict_proba(Xk.iloc[va])[:, 1] / len(SEEDS)
            models.append(m)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, yk)
    target = hard & ~known
    raw = np.mean([m.predict_proba(X[target])[:, 1] for m in models], axis=0)
    p = pd.Series(0.0, index=df.index)
    p[target] = iso.predict(raw)
    calib = pd.DataFrame({"oof_raw": oof, "oof_calibrated": iso.predict(oof), "is_unstable": yk})
    return p, calib


def choose_cut(n_sure: int, p: pd.Series) -> tuple[float, pd.DataFrame]:
    """Pick the probability cut that maximises expected F3 (ratio of expectations)."""
    cand = p[p > 0].sort_values(ascending=False)
    total = n_sure + cand.sum()
    rows = []
    for cut in sorted(cand.unique(), reverse=True):
        take = cand[cand >= cut]
        tp, flagged = n_sure + take.sum(), n_sure + len(take)
        rows.append({"cut": cut, "extra_flags": len(take), "expected_f3": 10 * tp / (9 * total + flagged)})
    rows.append({"cut": 1.01, "extra_flags": 0, "expected_f3": 10 * n_sure / (9 * total + n_sure)})
    table = pd.DataFrame(rows).sort_values("cut", ascending=False).reset_index(drop=True)
    # Flags at break-even only add variance: take the fewest flags within 1e-5 of the best.
    near_best = table[table.expected_f3 >= table.expected_f3.max() - 1e-5]
    best = near_best.loc[near_best.extra_flags.idxmin(), "cut"]
    return float(best), table


def plot_erratic(df: pd.DataFrame, label: pd.Series) -> None:
    mean = {g: df[cols(g)].mean(axis=1) for g in ERRATIC_BOX}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (gx, gy) in zip(axes, [("kda_std", "accuracy_std"), ("kda_std", "performance_stability")]):
        normal = label == "normal"
        ax.scatter(mean[gx][normal], mean[gy][normal], s=1, c="#9ca3af", alpha=0.3, label="normal")
        other = ~normal & (label != "erratic_combat")
        ax.scatter(mean[gx][other], mean[gy][other], s=2, c="#2563eb", alpha=0.5, label="other archetypes")
        err = label == "erratic_combat"
        ax.scatter(mean[gx][err], mean[gy][err], s=3, c="#dc2626", label="erratic combat")
        ax.set_xlabel(f"{gx} (mean of 4 periods)")
        ax.set_ylabel(f"{gy} (mean of 4 periods)")
    axes[0].legend(markerscale=6, loc="lower left")
    fig.suptitle("Erratic-combat archetype: a uniform box separate from the normal cloud")
    fig.savefig(ASSETS / "erratic-combat-box.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "test.csv")
    ids = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv", usecols=["id"])["id"]
    assert len(ids) == len(df)

    rules = archetype_rules(df)
    sure = rules.any(axis=1)
    label = pd.Series("normal", index=df.index)
    for c in rules.columns[::-1]:  # first matching rule wins
        label[rules[c]] = c

    X, hard = unstable_side_features(df)
    p, calib = unstable_nan_probability(df, X, hard)
    p[sure] = 0.0
    cut, cut_table = choose_cut(int(sure.sum()), p)
    extra = p >= cut
    label[extra] = "unstable_behavior_nan"
    flag = sure | extra

    print(rules.sum().to_string())
    print(f"rule flags {int(sure.sum())}, NaN-unstable candidates {int((p > 0).sum())} "
          f"(expected {p.sum():.1f} true), cut {cut:.3f} -> +{int(extra.sum())}, total {int(flag.sum())}")

    out = pd.DataFrame({"id": ids, "player_id": df.player_id, "archetype": label,
                        "p_unstable_nan": p.round(4), "is_anomaly": flag.astype(int)})
    out.to_csv(ROOT / "predictions" / "archetype-rules.csv", index=False)
    pd.DataFrame({"id": ids, "task5": flag.astype(int)}).to_csv(ROOT / "submit" / "index.csv", index=False)

    counts = label.value_counts().rename_axis("archetype").reset_index(name="players")
    counts["share_of_all_players"] = (counts.players / len(df)).round(4)
    counts.to_csv(ASSETS / "archetype-rule-counts.csv", index=False)
    cut_table.round(5).to_csv(ASSETS / "unstable-nan-cut-expected-f3.csv", index=False)
    bins = pd.cut(calib.oof_raw, [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], include_lowest=True)
    calib.groupby(bins, observed=True).agg(players=("is_unstable", "size"), unstable_rate=("is_unstable", "mean")) \
        .round(4).to_csv(ASSETS / "unstable-nan-calibration.csv")
    plot_erratic(df, label)


if __name__ == "__main__":
    main()
