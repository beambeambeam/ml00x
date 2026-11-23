"""Task 5: rule-based anomaly detection from the injected anomaly archetypes.

Normal players follow one pattern: each feature group is a personal base value
plus small period-to-period noise (period correlation 0.75-0.99). Anomalies are
injected as archetypes. Each archetype draws a few features from a uniform box
that is far from the normal range. These boxes have empty gaps to the normal
range, so exact rules separate them. Each rule fires on any observed signal,
which makes it robust to the ~5% NaN cells.

One archetype (unstable behaviour) has a single box feature,
`behavior_stability`. When that cell is NaN, a zero-parameter closed-form
analytical Bayesian posterior (`ExactBayesUB`) scores the player by integrating
the exact Normal vs. unstable longitudinal generative densities over
`purchase_frequency_1..4`, `login_count_1..4`, and `account_age_days`.
F3 gains from a flag when P(anomaly) > F3 / 10, so the cut maximises expected F3.

Task 5 has only test.csv. The rules and the Bayesian integral use it without
labels, which the competition allows for this task.

Usage (from task5/):
    uv run archetype_rules.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PERIODS = (1, 2, 3, 4)

# Exact 200-point Gauss-Legendre quadrature over Normal login_count base B_lc ~ Uniform(20, 100)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)
_LC_BASE_PTS = 0.5 * (100.0 - 20.0) * _GL_X + 0.5 * (100.0 + 20.0)
_LC_BASE_W = 0.5 * (100.0 - 20.0) * _GL_W / 80.0
_PF_NORM_C = np.array([0.50, 0.40, 0.30, 0.20], dtype=np.float64)
_LC_NORM_C = np.array([0.20, 0.30, 0.25, 0.15], dtype=np.float64)

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
        lc_std=lc.std(axis=1),
        lc_range=lc.max(axis=1) - lc.min(axis=1),
        pf_range=pf.max(axis=1) - pf.min(axis=1),
        age_gt_800=(df.account_age_days >= 800).astype(float),
        lc_norm_age=lc.mean(axis=1) / np.maximum(df.account_age_days, 1.0),
    )
    return X, hard


def marginal_mult_uniform(
    vals: np.ndarray, b_min: float, b_max: float, c_vec: np.ndarray
) -> np.ndarray:
    """Exact marginal density for X_p | B ~ Uniform(B*(1-c_p), B*(1+c_p)), B ~ Uniform(b_min, b_max)."""
    vals = np.asarray(vals, dtype=np.float64)
    obs = ~np.isnan(vals)
    k = obs.sum(axis=1)
    blo = np.full(len(vals), float(b_min), dtype=np.float64)
    bhi = np.full(len(vals), float(b_max), dtype=np.float64)
    log_denom = np.zeros(len(vals), dtype=np.float64)
    for p in range(4):
        m = obs[:, p]
        blo[m] = np.maximum(blo[m], vals[m, p] / (1.0 + c_vec[p]))
        bhi[m] = np.minimum(bhi[m], vals[m, p] / (1.0 - c_vec[p]))
        log_denom += np.where(m, np.log(2.0 * c_vec[p]), 0.0)
    valid = blo < bhi
    out = np.zeros(len(vals), dtype=np.float64)
    for i in np.flatnonzero(valid):
        ki = int(k[i])
        if ki == 0:
            integral = b_max - b_min
        elif ki == 1:
            integral = np.log(bhi[i] / blo[i])
        else:
            integral = (blo[i] ** (1 - ki) - bhi[i] ** (1 - ki)) / (ki - 1.0)
        out[i] = integral / ((b_max - b_min) * np.exp(log_denom[i]))
    return out


def prod_uniform_density_1d(
    z: np.ndarray, a1: float, b1: float, a2: float, b2: float
) -> np.ndarray:
    """Exact density of product U1 * U2 where U1 ~ Uniform(a1, b1), U2 ~ Uniform(a2, b2)."""
    lo = np.maximum(a1, z / b2)
    hi = np.minimum(b1, z / a2)
    mask = hi > lo
    safe_lo = np.where(mask, lo, 1.0)
    safe_hi = np.where(mask, hi, 1.0)
    return np.where(mask, np.log(safe_hi / safe_lo) / ((b1 - a1) * (b2 - a2)), 0.0)


def marginal_ub_lc_exact(
    vals: np.ndarray,
    c_norm: np.ndarray = _LC_NORM_C,
    m_lo: float = 0.40,
    m_hi: float = 0.70,
) -> np.ndarray:
    """1D Gauss-Legendre marginal likelihood of UB login_count under X_p = B*(1+u_p)*m_p."""
    vals = np.asarray(vals, dtype=np.float64)
    lik_grid = np.ones((len(vals), len(_LC_BASE_PTS)), dtype=np.float64)
    for p in range(4):
        vp = vals[:, p]
        obs = ~np.isnan(vp)
        if not np.any(obs):
            continue
        z = vp[obs, None] / _LC_BASE_PTS[None, :]
        fp = (
            prod_uniform_density_1d(z, 1.0 - c_norm[p], 1.0 + c_norm[p], m_lo, m_hi)
            / _LC_BASE_PTS[None, :]
        )
        lik_grid[obs] *= fp
    return lik_grid @ _LC_BASE_W


def unstable_nan_probability(df: pd.DataFrame, X: pd.DataFrame, hard: pd.Series):
    """Exact analytical Bayes P(unstable) for players with NaN behavior_stability."""
    del X  # Exact integral uses raw longitudinal periods directly from df
    other_7 = archetype_rules(df).drop(columns=["unstable_behavior"]).any(axis=1)
    known = df.behavior_stability.notna()
    known_clean = known & (~other_7)
    prior_ub = float((df.loc[known_clean, "behavior_stability"] < 0.55).sum() / known_clean.sum())

    pf_h = df.loc[hard, cols("purchase_frequency")].to_numpy(dtype=np.float64)
    lc_h = df.loc[hard, cols("login_count")].to_numpy(dtype=np.float64)
    age_h = df.loc[hard, "account_age_days"].to_numpy(dtype=np.float64)

    p_pf_norm = marginal_mult_uniform(pf_h, 0.0, 5.0, _PF_NORM_C)
    p_lc_norm = marginal_mult_uniform(lc_h, 20.0, 100.0, _LC_NORM_C)
    p_age_norm = np.where(
        np.isnan(age_h),
        1.0,
        np.where((age_h >= 180.0) & (age_h <= 1500.0), 1.0 / 1320.0, 0.0),
    )
    l_norm = p_pf_norm * p_lc_norm * p_age_norm

    k_pf = (~np.isnan(pf_h)).sum(axis=1)
    in_pf_ub = np.all(np.isnan(pf_h) | ((pf_h >= 3.0) & (pf_h <= 6.0)), axis=1)
    p_pf_ub = np.where(in_pf_ub, (1.0 / 3.0) ** k_pf, 0.0)
    p_lc_ub = marginal_ub_lc_exact(lc_h, c_norm=_LC_NORM_C, m_lo=0.40, m_hi=0.70)
    p_age_ub = np.where(
        np.isnan(age_h),
        1.0,
        np.where((age_h >= 800.0) & (age_h <= 1500.0), 1.0 / 700.0, 0.0),
    )
    l_ub = p_pf_ub * p_lc_ub * p_age_ub

    num = prior_ub * l_ub
    den = num + (1.0 - prior_ub) * l_norm
    post_h = np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)

    post_full = pd.Series(0.0, index=df.index)
    post_full.loc[hard] = post_h

    target = hard & ~known
    p = pd.Series(0.0, index=df.index)
    p.loc[target] = post_full.loc[target]

    hk = hard & known_clean
    yk = (df.loc[hk, "behavior_stability"] < 0.55).astype(int).to_numpy()
    post_hk = post_full.loc[hk].to_numpy()
    calib = pd.DataFrame({"oof_raw": post_hk, "oof_calibrated": post_hk, "is_unstable": yk})
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
