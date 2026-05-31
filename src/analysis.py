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
    # Always use the `group` column values (control/treatment) rather
    # than attempting to detect a separate `control` column.
    mismatch_mask = (
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
    ctrl_mean = ctrl.mean()
    rel_lift = None
    if ctrl_mean != 0:
        rel_lift = round((treat.mean() - ctrl_mean) / ctrl_mean, 4)

    return {
        "control_rate":   round(ctrl_mean, 4),
        "treatment_rate": round(treat.mean(), 4),
        "absolute_lift":  round(treat.mean() - ctrl_mean, 4),
        "relative_lift":  rel_lift,
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
    # Guard against empty groups which would cause division by zero.
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan)

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


# ── Subgroup / heterogeneity analysis ─────────────────────────────────────────

def subgroup_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lift, CI, and significance for each time-based subgroup.

    Subgroups:
      hour_bucket  — Night (0–5), Morning (6–11), Afternoon (12–17), Evening (18–23)
      day_type     — Weekday (Mon–Fri) vs. Weekend (Sat–Sun)

    Returns a DataFrame with one row per subgroup. P-values are Bonferroni-
    corrected for the number of tests performed.
    """
    df = add_time_features(df.copy())

    df["hour_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (0–5)", "Morning (6–11)", "Afternoon (12–17)", "Evening (18–23)"],
    )
    df["day_type"] = df["day_of_week"].apply(
        lambda d: "Weekend" if d >= 5 else "Weekday"
    )

    rows = []
    for col in ("hour_bucket", "day_type"):
        for val in sorted(df[col].dropna().unique(), key=str):
            subset = df[df[col] == val]
            ctrl  = subset[subset["group"] == "control"]["converted"]
            treat = subset[subset["group"] == "treatment"]["converted"]

            if len(ctrl) < 50 or len(treat) < 50:
                continue

            count = np.array([treat.sum(), ctrl.sum()])
            nobs  = np.array([len(treat), len(ctrl)])
            _, p_raw = proportions_ztest(count, nobs)
            ci_lo, ci_hi = proportion_confint_diff(treat, ctrl)

            rows.append({
                "subgroup_type":  col,
                "subgroup":       str(val),
                "n_control":      len(ctrl),
                "n_treatment":    len(treat),
                "control_rate":   round(ctrl.mean(),  4),
                "treatment_rate": round(treat.mean(), 4),
                "lift":           round(treat.mean() - ctrl.mean(), 4),
                "ci_low":         round(ci_lo, 4),
                "ci_high":        round(ci_hi, 4),
                "p_raw":          round(float(p_raw), 4),
            })

    result = pd.DataFrame(rows)
    n_tests = len(result)
    result["p_bonferroni"] = (result["p_raw"] * n_tests).clip(upper=1.0).round(4)
    result["significant"]  = result["p_bonferroni"] < 0.05
    return result


# ── CUPED (variance reduction) ────────────────────────────────────────────────

def cuped_adjustment(df: pd.DataFrame) -> dict:
    """
    CUPED — Controlled-experiment Using Pre-Experiment Data.
    Formula: Y_cuped = Y - θ * (X - X̄),  θ = cov(Y_ctrl, X_ctrl) / var(X_ctrl)
    Covariate X = hour_of_day (proxy; no true pre-experiment data in this dataset).
    θ is estimated on the control group only to avoid treatment-arm leakage.
    """
    from scipy.stats import ttest_ind

    df = add_time_features(df.copy())
    ctrl = df[df["group"] == "control"]

    cov_matrix = np.cov(ctrl["converted"].values, ctrl["hour"].values)
    theta = cov_matrix[0, 1] / cov_matrix[1, 1]

    x_bar = df["hour"].mean()
    df["converted_cuped"] = df["converted"] - theta * (df["hour"] - x_bar)

    ctrl_orig  = df[df["group"] == "control"]["converted"]
    treat_orig = df[df["group"] == "treatment"]["converted"]
    ctrl_adj   = df[df["group"] == "control"]["converted_cuped"]
    treat_adj  = df[df["group"] == "treatment"]["converted_cuped"]

    var_orig = float(ctrl_orig.var())
    var_adj  = float(ctrl_adj.var())
    var_reduction_pct = (1 - var_adj / var_orig) * 100 if var_orig > 0 else 0.0

    se_orig = float(np.sqrt(var_orig / len(ctrl_orig) + treat_orig.var() / len(treat_orig)))
    se_adj  = float(np.sqrt(var_adj  / len(ctrl_adj)  + treat_adj.var()  / len(treat_adj)))

    stat, p = ttest_ind(treat_adj.values, ctrl_adj.values)

    return {
        "theta":              round(theta, 8),
        "x_bar":              round(float(x_bar), 4),
        "var_orig":           round(var_orig, 8),
        "var_adj":            round(var_adj, 8),
        "var_reduction_pct":  round(var_reduction_pct, 3),
        "se_orig":            round(se_orig, 8),
        "se_adj":             round(se_adj, 8),
        "se_reduction_pct":   round((1 - se_adj / se_orig) * 100 if se_orig > 0 else 0.0, 3),
        "mean_ctrl_orig":     round(float(ctrl_orig.mean()), 6),
        "mean_treat_orig":    round(float(treat_orig.mean()), 6),
        "mean_ctrl_adj":      round(float(ctrl_adj.mean()), 6),
        "mean_treat_adj":     round(float(treat_adj.mean()), 6),
        "lift_orig":          round(float(treat_orig.mean() - ctrl_orig.mean()), 6),
        "lift_adj":           round(float(treat_adj.mean() - ctrl_adj.mean()), 6),
        "t_stat":             round(float(stat), 4),
        "p_value":            round(float(p), 4),
        "significant":        bool(p < 0.05),
        "n_control":          len(ctrl_orig),
        "n_treatment":        len(treat_orig),
    }


def heterogeneity_test(df: pd.DataFrame) -> dict:
    """
    Logistic regression interaction test: does the treatment effect differ
    by hour bucket? Returns the likelihood-ratio test p-value comparing
    a model with interaction terms to one without.
    """
    import statsmodels.formula.api as smf

    df = add_time_features(df.copy())
    df["hour_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    )
    df["treatment"] = (df["group"] == "treatment").astype(int)

    base        = smf.logit("converted ~ treatment + C(hour_bucket)", data=df).fit(disp=False)
    interaction = smf.logit("converted ~ treatment * C(hour_bucket)", data=df).fit(disp=False)

    from scipy.stats import chi2
    lr_stat = 2 * (interaction.llf - base.llf)
    df_diff = interaction.df_model - base.df_model
    p_value = float(chi2.sf(lr_stat, df_diff))

    return {
        "lr_stat":          round(lr_stat, 4),
        "df":               int(df_diff),
        "p_value":          round(p_value, 4),
        "heterogeneity":    p_value < 0.05,
    }
