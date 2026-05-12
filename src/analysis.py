"""
Core analysis helpers for the A/B test causal inference project.
Dataset: Udacity A/B Test Results (website redesign conversion study).
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower


# ── Data loading & cleaning ───────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def audit_data(df: pd.DataFrame) -> dict:
    """Return a dict of data quality findings."""
    n_total = len(df)
    n_dupes = df["user_id"].duplicated().sum()

    # Mismatches: treatment group shown old page, or control shown new page
    mismatch_mask = (
        ((df["group"] == "treatment") & (df["landing_page"] == "old_page")) |
        ((df["control"] == "control") & (df["landing_page"] == "new_page"))
        if "control" in df.columns
        else
        ((df["group"] == "treatment") & (df["landing_page"] == "old_page")) |
        ((df["group"] == "control") & (df["landing_page"] == "new_page"))
    )
    n_mismatches = mismatch_mask.sum()

    return {
        "n_total": n_total,
        "n_duplicates": int(n_dupes),
        "n_mismatches": int(n_mismatches),
        "mismatch_rate": round(n_mismatches / n_total, 4),
        "conversion_overall": round(df["converted"].mean(), 4),
        "group_balance": df["group"].value_counts().to_dict(),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate users (keep first) and page/group mismatches."""
    df = df.drop_duplicates(subset="user_id", keep="first").copy()
    mismatch = (
        ((df["group"] == "treatment") & (df["landing_page"] == "old_page")) |
        ((df["group"] == "control") & (df["landing_page"] == "new_page"))
    )
    return df[~mismatch].reset_index(drop=True)


# ── Naive analysis ────────────────────────────────────────────────────────────

def naive_comparison(df: pd.DataFrame) -> dict:
    """Simple conversion rate comparison — the wrong way to analyse an A/B test."""
    ctrl = df[df["group"] == "control"]["converted"]
    treat = df[df["group"] == "treatment"]["converted"]
    return {
        "control_rate":   round(ctrl.mean(), 4),
        "treatment_rate": round(treat.mean(), 4),
        "absolute_lift":  round(treat.mean() - ctrl.mean(), 4),
        "relative_lift":  round((treat.mean() - ctrl.mean()) / ctrl.mean(), 4),
        "n_control":      len(ctrl),
        "n_treatment":    len(treat),
    }


# ── Statistical testing ───────────────────────────────────────────────────────

def ztest_conversion(df: pd.DataFrame) -> dict:
    """Two-proportion z-test for conversion rate difference."""
    ctrl  = df[df["group"] == "control"]["converted"]
    treat = df[df["group"] == "treatment"]["converted"]

    count = np.array([treat.sum(), ctrl.sum()])
    nobs  = np.array([len(treat), len(ctrl)])

    stat, p_value = proportions_ztest(count, nobs)
    ci_low, ci_high = proportion_confint_diff(treat, ctrl)

    return {
        "z_stat":      round(float(stat), 4),
        "p_value":     round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
        "ci_95":       (round(ci_low, 4), round(ci_high, 4)),
    }


def proportion_confint_diff(treat: pd.Series, ctrl: pd.Series, alpha: float = 0.05) -> tuple:
    """95% CI on the difference in proportions (treatment - control)."""
    p1, p2 = treat.mean(), ctrl.mean()
    n1, n2 = len(treat), len(ctrl)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p1 - p2
    return diff - z * se, diff + z * se


# ── Power analysis ────────────────────────────────────────────────────────────

def power_analysis(df: pd.DataFrame, alpha: float = 0.05, desired_power: float = 0.80) -> dict:
    """How large a sample do we actually need to detect the observed effect?"""
    naive = naive_comparison(df)
    p_ctrl  = naive["control_rate"]
    p_treat = naive["treatment_rate"]

    effect_size = proportion_effectsize(p_treat, p_ctrl)
    analysis    = NormalIndPower()
    required_n  = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=desired_power,
        alternative="two-sided",
    )

    actual_power = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        nobs1=naive["n_control"],
        alternative="two-sided",
    )

    return {
        "effect_size":    round(effect_size, 4),
        "required_n_per_group": int(np.ceil(required_n)),
        "actual_n_per_group":   naive["n_control"],
        "actual_power":         round(float(actual_power), 4),
        "adequately_powered":   naive["n_control"] >= required_n,
    }


# ── Propensity score matching ─────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and day-of-week as covariates for PSM."""
    df = df.copy()
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def propensity_score_match(df: pd.DataFrame, caliper: float | None = None) -> pd.DataFrame:
    """
    Estimate propensity scores with logistic regression on time covariates,
    then 1:1 nearest-neighbour match within caliper.

    Caliper defaults to the Rubin (2001) rule: 0.2 × SD(logit(propensity)).
    In a well-randomised experiment all scores cluster near 0.5, so the
    matched sample will be small — that is expected and is itself a finding.
    Returns a matched DataFrame with equal control/treatment sizes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = add_time_features(df)
    features = ["hour", "day_of_week"]

    X = df[features].values
    y = (df["group"] == "treatment").astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_scaled, y)
    df = df.copy()
    ps = lr.predict_proba(X_scaled)[:, 1]
    df["propensity"] = ps

    if caliper is None:
        # Rubin rule: 0.2 × SD of logit-transformed propensity scores
        logit_ps = np.log(ps / (1 - np.clip(ps, 1e-6, 1 - 1e-6)))
        caliper  = 0.2 * float(np.std(logit_ps))

    treat_idx = df.index[df["group"] == "treatment"].tolist()
    ctrl_idx  = df.index[df["group"] == "control"].tolist()

    treat_scores = df.loc[treat_idx, "propensity"].values
    ctrl_scores  = df.loc[ctrl_idx,  "propensity"].values

    matched_treat, matched_ctrl = [], []
    used_ctrl = set()

    for ti, ts in zip(treat_idx, treat_scores):
        diffs = np.abs(ctrl_scores - ts)
        best  = int(np.argmin(diffs))
        if diffs[best] <= caliper and ctrl_idx[best] not in used_ctrl:
            matched_treat.append(ti)
            matched_ctrl.append(ctrl_idx[best])
            used_ctrl.add(ctrl_idx[best])

    matched_df = pd.concat([
        df.loc[matched_treat],
        df.loc[matched_ctrl],
    ]).reset_index(drop=True)

    return matched_df
