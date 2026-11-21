"""Train and validate the Task 1 account-cohort classifier (experiment 005)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "005"
PREDICTIONS = ROOT / "predictions" / "experiments"
TARGET = "is_cheater"
BLEND_WEIGHT = 0.4
EPSILON = 1e-5
ACCOUNT_FIELDS = ("account_age_days", "level", "level_progression_speed")
SUPPORT_FIELDS = (
    *ACCOUNT_FIELDS[:2],
    "friend_network_size",
    "reports_received",
    "crosshair_placement",
)
SKILL_FIELDS = (
    "kill_death_ratio",
    "headshot_percentage",
    "win_rate",
    "accuracy_score",
    "kill_consistency",
    "reaction_time_ms",
    "aiming_smoothness",
    "spray_control_score",
    "game_sense_score",
    "input_consistency_score",
    "movement_pattern_score",
    "buy_decision_score",
    "first_blood_rate",
    "weapon_switch_speed",
)
# Each row gives the label and inclusive ranges for age, friends, and reports.
# The ranges were inferred from the labeled file and checked for contradictions per fold.
COHORTS = (
    (0, (1, 110), (0, 49), (10, 29)),
    (0, (120, 900), (45, 249), (5, 19)),
    (0, (350, 1300), (120, 349), (8, 24)),
    (0, (600, 1500), (100, 319), (5, 17)),
    (1, (15, 180), (10, 89), (4, 15)),
    (1, (15, 180), (10, 89), (6, 19)),
    (1, (200, 1000), (60, 269), (4, 15)),
    (1, (200, 1000), (60, 269), (6, 19)),
)
COHORT_LABELS = np.array([label for label, *_ in COHORTS])


def uniform_density(values, low, high, *, discrete=False):
    """Return one for missing observations, which add no independent evidence."""
    return np.where(
        np.isnan(values), 1.0, ((low <= values) & (values <= high)) / (high - low + discrete)
    )


def account_density(values, age_range, level_range):
    """Marginalize progression = level / age × Uniform(0.15, 1.08)."""
    age, level, speed = values.T
    age_min, age_max = age_range
    level_min, level_max = level_range
    mult_min, mult_max = 0.15, 1.08
    age_width = age_max - age_min
    level_width = level_max - level_min
    mult_width = mult_max - mult_min
    has_age, has_level, has_speed = np.isfinite(values).T
    density = uniform_density(age, *age_range) * uniform_density(level, *level_range)
    density[has_speed & (speed <= 0)] = 0
    has_speed &= speed > 0

    complete = has_age & has_level & has_speed
    ratio = age[complete] / np.maximum(level[complete], level_min)
    density[complete] *= uniform_density(speed[complete] * ratio, mult_min, mult_max) * ratio

    missing_age = ~has_age & has_level & has_speed
    level_seen = np.maximum(level[missing_age], level_min)
    age_max_ratio = np.minimum(age_max, mult_max * level_seen / speed[missing_age])
    age_min_ratio = np.maximum(age_min, mult_min * level_seen / speed[missing_age])
    density[missing_age] *= np.maximum(age_max_ratio**2 - age_min_ratio**2, 0) / (
        2 * age_width * mult_width * level_seen
    )

    missing_level = has_age & ~has_level & has_speed
    age_seen = np.maximum(age[missing_level], age_min)
    speed_seen = speed[missing_level]
    upper_level = np.minimum(level_max, speed_seen * age_seen / mult_min)
    lower_level = np.maximum(level_min, speed_seen * age_seen / mult_max)
    density[missing_level] *= (
        np.maximum(np.log(upper_level / lower_level), 0) * age_seen / (level_width * mult_width)
    )

    speed_only = ~has_age & ~has_level & has_speed
    speed_seen = speed[speed_only]
    # Integrate over level, splitting at the points where either age bound starts to bind.
    breakpoints = [np.full_like(speed_seen, edge) for edge in level_range]
    for age_bound in age_range:
        for multiplier_bound in (mult_min, mult_max):
            breakpoints.append(np.clip(speed_seen * age_bound / multiplier_bound, *level_range))
    edges = np.sort(np.column_stack(breakpoints), axis=1)
    half_width = np.diff(edges, axis=1) / 2
    midpoint = edges[:, :-1] + half_width
    nodes, weights = np.polynomial.legendre.leggauss(12)
    level_nodes = midpoint[..., None] + half_width[..., None] * nodes
    lower_age = np.maximum(age_min, mult_min * level_nodes / speed_seen[:, None, None])
    upper_age = np.minimum(age_max, mult_max * level_nodes / speed_seen[:, None, None])
    integrand = np.maximum(upper_age**2 - lower_age**2, 0) / (
        2 * age_width * level_width * mult_width * level_nodes
    )
    density[speed_only] *= ((integrand @ weights) * half_width).sum(axis=1)
    return density


def cohort_likelihoods(frame):
    account = frame[list(ACCOUNT_FIELDS)].to_numpy(dtype=float)
    friends = frame["friend_network_size"].to_numpy(dtype=float)
    reports = frame["reports_received"].to_numpy(dtype=float)
    crosshair = frame["crosshair_placement"].to_numpy(dtype=float)
    columns = []
    for label, age_range, friend_range, report_range in COHORTS:
        level_range = (25, 96) if label else (28, 100)
        crosshair_range = (0.80, 0.94) if label else (0.82, 0.96)
        columns.append(
            account_density(account, age_range, level_range)
            * uniform_density(friends, *friend_range, discrete=True)
            * uniform_density(reports, *report_range, discrete=True)
            * uniform_density(crosshair, *crosshair_range)
        )
    return np.column_stack(columns)


def cohort_weights(likelihoods, labels):
    compatible = likelihoods * (labels[:, None] == COHORT_LABELS)
    if not compatible.any(axis=1).all():
        raise ValueError("A labeled row falls outside every cohort of its class")
    weights = np.full(len(COHORTS), 1 / len(COHORTS))
    for _ in range(100):
        joint = compatible * weights
        updated = (joint / joint.sum(axis=1, keepdims=True)).mean(axis=0)
        if np.max(np.abs(updated - weights)) < 1e-9:
            return updated
        weights = updated
    raise RuntimeError("Cohort weight estimation did not converge")


def design_matrix(likelihoods, priors, skill_scores):
    joint = likelihoods * priors
    total = joint.sum(axis=1, keepdims=True)
    membership = np.divide(joint, total, out=np.zeros_like(joint), where=total > 0)
    legit = membership[:, COHORT_LABELS == 0].sum(axis=1)
    cheater = membership[:, COHORT_LABELS == 1].sum(axis=1)
    cheater[total[:, 0] == 0] = 0.5
    # Shared skill terms plus deviations for cohorts with different skill distributions.
    interactions = np.column_stack([np.ones(len(legit)), membership[:, [0, 2, 3, 5, 7]], legit])
    linear = (interactions[:, :, None] * skill_scores[:, None, :]).reshape(len(legit), -1)
    quadratic = (interactions[:, :, None] * skill_scores[:, None, :] ** 2).reshape(len(legit), -1)
    prior_log_odds = logit(cheater.clip(EPSILON, 1 - EPSILON))
    return np.column_stack([prior_log_odds, interactions[:, 1:], linear, quadratic])


def skill_scores(transformer, skills):
    return np.nan_to_num(transformer.transform(skills).clip(-3.3, 3.3))


def fit_and_predict(train_likelihoods, train_skills, labels, eval_likelihoods, eval_skills):
    priors = cohort_weights(train_likelihoods, labels)
    transformer = QuantileTransformer(
        n_quantiles=300, output_distribution="normal", subsample=None
    ).fit(train_skills)
    train = design_matrix(train_likelihoods, priors, skill_scores(transformer, train_skills))
    model = LogisticRegression(C=0.04, max_iter=600).fit(train, labels)
    if model.n_iter_[0] == model.max_iter:
        raise RuntimeError("Logistic regression did not converge")
    design = design_matrix(eval_likelihoods, priors, skill_scores(transformer, eval_skills))
    return model.predict_proba(design)[:, 1], priors


def apply_support(probabilities, likelihoods):
    legit = likelihoods[:, COHORT_LABELS == 0].any(axis=1)
    cheater = likelihoods[:, COHORT_LABELS == 1].any(axis=1)
    scores = np.where(legit | cheater, probabilities, np.nan)
    return np.where(legit & ~cheater, EPSILON, np.where(cheater & ~legit, 1 - EPSILON, scores))


def best_threshold(y, probability):
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    ends = np.flatnonzero(np.r_[np.diff(sorted_probability) != 0, True])
    scores = 5 * np.cumsum(y[order])[ends] / (4 * y.sum() + ends + 1)
    return float(sorted_probability[ends[np.argmax(scores)]])


def f2(y, prediction):
    return float(5 * (y & prediction).sum() / (4 * y.sum() + prediction.sum()))


def evaluate(y, probability, folds, complete):
    thresholds = {
        fold: best_threshold(y[folds != fold], probability[folds != fold])
        for fold in np.unique(folds)
    }
    prediction = probability >= np.array([thresholds[fold] for fold in folds])
    metrics = {
        "f2": f2(y, prediction),
        "f2_complete": f2(y[complete], prediction[complete]),
        "f2_incomplete": f2(y[~complete], prediction[~complete]),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "log_loss": log_loss(y, probability),
    }
    return metrics, prediction, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote", action="store_true", help="Update submit/index.csv if all folds improve"
    )
    args = parser.parse_args()

    train = pd.read_csv(ROOT / "data" / "train.csv").dropna(subset=[TARGET]).reset_index(drop=True)
    baseline = pd.read_csv(PREDICTIONS / "004" / "oof.csv")
    if (
        not train[["player_id", TARGET]]
        .astype({TARGET: int})
        .equals(baseline[["player_id", TARGET]])
    ):
        raise ValueError("Experiment 004 OOF rows and labels do not match the training file")
    labels = train[TARGET].to_numpy(dtype=bool)
    folds = baseline["fold"].to_numpy()
    oof_baseline = baseline["proba_exp004"].to_numpy()
    likelihoods = cohort_likelihoods(train)
    skills = train[list(SKILL_FIELDS)].to_numpy()
    complete = train[list(SUPPORT_FIELDS)].notna().all(axis=1).to_numpy()
    oof = np.empty(len(labels))
    for fold in np.unique(folds):
        valid = folds == fold
        oof[valid], _ = fit_and_predict(
            likelihoods[~valid], skills[~valid], labels[~valid], likelihoods[valid], skills[valid]
        )

    oof_support_only = apply_support(oof_baseline, likelihoods)
    oof_final = apply_support((1 - BLEND_WEIGHT) * oof_baseline + BLEND_WEIGHT * oof, likelihoods)
    base_metrics, base_preds, _ = evaluate(labels, oof_baseline, folds, complete)
    support_metrics, _, _ = evaluate(labels, oof_support_only, folds, complete)
    metrics, predictions, thresholds = evaluate(labels, oof_final, folds, complete)
    per_fold = []
    for fold, threshold in thresholds.items():
        mask = folds == fold
        old = f2(labels[mask], base_preds[mask])
        new = f2(labels[mask], predictions[mask])
        per_fold.append(
            {
                "fold": int(fold),
                "baseline_f2": old,
                "f2": new,
                "delta_f2": new - old,
                "threshold": threshold,
            }
        )
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(1000):
        indices = rng.integers(len(labels), size=len(labels))
        differences.append(
            f2(labels[indices], predictions[indices]) - f2(labels[indices], base_preds[indices])
        )
    interval = np.quantile(differences, [0.025, 0.975]).tolist()
    threshold = best_threshold(labels, oof_final)
    promote = interval[0] > 0 and all(row["delta_f2"] > 0 for row in per_fold)
    print(
        pd.DataFrame(
            [base_metrics, support_metrics, metrics], index=["exp004", "support_only", "exp005"]
        ).to_string()
    )
    print(pd.DataFrame(per_fold).to_string(index=False))
    print(f"Paired bootstrap 95% interval: {interval}; all-OOF threshold: {threshold:.6f}")
    if args.promote and not promote:
        parser.error("The candidate did not pass validation for promotion")

    test = pd.read_csv(ROOT / "data" / "test.csv")
    test_baseline = pd.read_csv(PREDICTIONS / "004" / "test.csv")
    if not test["player_id"].equals(test_baseline["player_id"]):
        raise ValueError("Experiment 004 prediction rows do not match the test file")
    test_likelihoods = cohort_likelihoods(test)
    test_probability, priors = fit_and_predict(
        likelihoods, skills, labels, test_likelihoods, test[list(SKILL_FIELDS)].to_numpy()
    )
    original = test_baseline["proba"].to_numpy()
    blended = (1 - BLEND_WEIGHT) * original + BLEND_WEIGHT * test_probability
    final = apply_support(blended, test_likelihoods)
    # Leave unsupported test rows to the existing classifier and its existing cutoff.
    unsupported = np.isnan(final)
    final[unsupported] = original[unsupported]
    with (ROOT / "models" / "experiments" / "004" / "stack_meta.json").open() as file:
        old_threshold = json.load(file)["threshold"]
    prediction = final >= np.where(unsupported, old_threshold, threshold)
    sample = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv", usecols=["id"])
    if len(sample) != len(test) or not sample["id"].is_unique:
        raise ValueError("Official sample submission does not match the test row count")
    submission = sample.assign(task1=prediction.astype(int))

    assets = ROOT / "assets" / "experiments" / EXPERIMENT
    artifacts = ROOT / "models" / "experiments" / EXPERIMENT
    output = PREDICTIONS / EXPERIMENT
    for directory in (assets, artifacts, output):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [base_metrics, support_metrics, metrics], index=["exp004", "support_only", "exp005"]
    ).to_csv(assets / "results.csv", index_label="model")
    pd.DataFrame(per_fold).to_csv(assets / "per-fold-f2.csv", index=False)
    baseline[["player_id", "fold", TARGET]].assign(
        proba_exp005=oof_final, prediction=predictions.astype(int)
    ).to_csv(output / "oof.csv", index=False)
    test[["player_id"]].assign(proba=final, prediction=prediction.astype(int)).to_csv(
        output / "test.csv", index=False
    )
    metrics_summary = {
        "experiment": EXPERIMENT,
        "cohort_weights": priors.tolist(),
        "blend_weight": BLEND_WEIGHT,
        "threshold": threshold,
        "metrics_exp004": base_metrics,
        "metrics_exp005": metrics,
        "paired_bootstrap_delta_f2_95_percent": interval,
        "test_rows_outside_all_cohorts": int(unsupported.sum()),
        "passed_promotion_gate": promote,
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics_summary, indent=2) + "\n")
    if args.promote:
        submission.to_csv(ROOT / "submit" / "index.csv", index=False)


if __name__ == "__main__":
    main()
