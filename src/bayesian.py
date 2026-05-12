"""
Bayesian A/B testing using the Beta-Binomial conjugate model.

Model:
  Prior:     theta ~ Beta(alpha_0, beta_0)   [default: uniform Beta(1,1)]
  Likelihood: k | theta, n ~ Binomial(n, theta)
  Posterior: theta | k, n ~ Beta(alpha_0 + k, beta_0 + n - k)

Key outputs:
  - Posterior distributions for control and treatment
  - P(treatment > control)  — probability of superiority
  - Expected loss for each ship decision
  - 95% highest-density credible intervals
"""

import numpy as np
import pandas as pd
from scipy import stats


def beta_posterior(
    n_conversions: int,
    n_total: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> tuple[float, float]:
    """Return (alpha_post, beta_post) for a Beta-Binomial posterior."""
    return alpha_prior + n_conversions, beta_prior + (n_total - n_conversions)


def prob_treatment_better(
    alpha_ctrl: float,
    beta_ctrl: float,
    alpha_treat: float,
    beta_treat: float,
    n_samples: int = 200_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo estimate of P(theta_treat > theta_ctrl).

    With ~145k observations per group the posteriors are very tight, so
    200k samples give a stable estimate to 4 decimal places.
    """
    rng = np.random.default_rng(seed)
    ctrl_samples  = rng.beta(alpha_ctrl,  beta_ctrl,  size=n_samples)
    treat_samples = rng.beta(alpha_treat, beta_treat, size=n_samples)
    return float((treat_samples > ctrl_samples).mean())


def expected_loss(
    alpha_ctrl: float,
    beta_ctrl: float,
    alpha_treat: float,
    beta_treat: float,
    n_samples: int = 200_000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Expected loss (regret) for each decision:

    loss_if_ship_treatment = E[max(theta_ctrl - theta_treat, 0)]
        Risk of choosing treatment when control is actually better.

    loss_if_keep_control   = E[max(theta_treat - theta_ctrl, 0)]
        Opportunity cost of keeping control when treatment is actually better.
    """
    rng = np.random.default_rng(seed)
    ctrl_samples  = rng.beta(alpha_ctrl,  beta_ctrl,  size=n_samples)
    treat_samples = rng.beta(alpha_treat, beta_treat, size=n_samples)
    return {
        "loss_if_ship_treatment": float(np.maximum(ctrl_samples - treat_samples, 0).mean()),
        "loss_if_keep_control":   float(np.maximum(treat_samples - ctrl_samples, 0).mean()),
    }


def credible_interval(
    alpha_post: float,
    beta_post: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Equal-tailed credible interval for a Beta posterior."""
    tail = (1 - level) / 2
    return (
        float(stats.beta.ppf(tail,     alpha_post, beta_post)),
        float(stats.beta.ppf(1 - tail, alpha_post, beta_post)),
    )


def bayesian_ab_summary(
    df: pd.DataFrame,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict:
    """
    Full Bayesian summary for a cleaned A/B test DataFrame.

    Returns posterior parameters, point estimates, credible intervals,
    P(treatment > control), and expected loss for both decisions.
    """
    ctrl  = df[df["group"] == "control"]["converted"]
    treat = df[df["group"] == "treatment"]["converted"]

    a_ctrl,  b_ctrl  = beta_posterior(int(ctrl.sum()),  len(ctrl),  alpha_prior, beta_prior)
    a_treat, b_treat = beta_posterior(int(treat.sum()), len(treat), alpha_prior, beta_prior)

    p_better = prob_treatment_better(a_ctrl, b_ctrl, a_treat, b_treat)
    losses   = expected_loss(a_ctrl, b_ctrl, a_treat, b_treat)

    return {
        "prior":                  (alpha_prior, beta_prior),
        "posterior_control":      (a_ctrl,  b_ctrl),
        "posterior_treatment":    (a_treat, b_treat),
        "mean_control":           round(a_ctrl  / (a_ctrl  + b_ctrl),  6),
        "mean_treatment":         round(a_treat / (a_treat + b_treat), 6),
        "ci_95_control":          credible_interval(a_ctrl,  b_ctrl),
        "ci_95_treatment":        credible_interval(a_treat, b_treat),
        "prob_treatment_better":  round(p_better,       4),
        "prob_control_better":    round(1 - p_better,   4),
        **{k: round(v, 6) for k, v in losses.items()},
    }
