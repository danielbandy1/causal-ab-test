"""Tests for src/bayesian.py."""

import numpy as np
import pandas as pd
import pytest

from src.bayesian import (
    beta_posterior,
    prob_treatment_better,
    expected_loss,
    credible_interval,
    bayesian_ab_summary,
)


# ── beta_posterior ────────────────────────────────────────────────────────────

def test_beta_posterior_uniform_prior():
    a, b = beta_posterior(n_conversions=10, n_total=100)
    assert a == pytest.approx(11.0)
    assert b == pytest.approx(91.0)


def test_beta_posterior_informative_prior():
    a, b = beta_posterior(n_conversions=10, n_total=100, alpha_prior=5, beta_prior=5)
    assert a == pytest.approx(15.0)
    assert b == pytest.approx(95.0)


def test_beta_posterior_zero_conversions():
    a, b = beta_posterior(n_conversions=0, n_total=50)
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(51.0)


def test_beta_posterior_all_conversions():
    a, b = beta_posterior(n_conversions=50, n_total=50)
    assert a == pytest.approx(51.0)
    assert b == pytest.approx(1.0)


# ── prob_treatment_better ─────────────────────────────────────────────────────

def test_prob_treatment_better_range():
    p = prob_treatment_better(100, 900, 120, 880)
    assert 0.0 <= p <= 1.0


def test_prob_treatment_better_clearly_better():
    """Treatment with 20% rate vs control at 10% — should be > 0.99."""
    a_ctrl,  b_ctrl  = beta_posterior(100, 1_000)  # 10%
    a_treat, b_treat = beta_posterior(200, 1_000)  # 20%
    p = prob_treatment_better(a_ctrl, b_ctrl, a_treat, b_treat)
    assert p > 0.99


def test_prob_treatment_better_clearly_worse():
    """Treatment with 5% rate vs control at 15% — should be < 0.01."""
    a_ctrl,  b_ctrl  = beta_posterior(150, 1_000)  # 15%
    a_treat, b_treat = beta_posterior(50,  1_000)  # 5%
    p = prob_treatment_better(a_ctrl, b_ctrl, a_treat, b_treat)
    assert p < 0.01


def test_prob_treatment_better_near_half_when_equal():
    """Identical rates should give P ≈ 0.5."""
    a, b = beta_posterior(120, 1_000)
    p = prob_treatment_better(a, b, a, b, seed=0)
    assert 0.45 <= p <= 0.55


# ── expected_loss ─────────────────────────────────────────────────────────────

def test_expected_loss_keys():
    result = expected_loss(100, 900, 110, 890)
    assert "loss_if_ship_treatment" in result
    assert "loss_if_keep_control"   in result


def test_expected_loss_nonnegative():
    result = expected_loss(100, 900, 110, 890)
    assert result["loss_if_ship_treatment"] >= 0
    assert result["loss_if_keep_control"]   >= 0


def test_expected_loss_dominant_treatment():
    """When treatment is clearly better, loss of keeping control >> loss of shipping treatment."""
    a_ctrl,  b_ctrl  = beta_posterior(100, 1_000)   # 10%
    a_treat, b_treat = beta_posterior(200, 1_000)   # 20%
    result = expected_loss(a_ctrl, b_ctrl, a_treat, b_treat)
    assert result["loss_if_keep_control"] > result["loss_if_ship_treatment"] * 5


# ── credible_interval ─────────────────────────────────────────────────────────

def test_credible_interval_ordered():
    lo, hi = credible_interval(50, 950)
    assert lo < hi


def test_credible_interval_contains_mean():
    alpha, beta = 120, 880   # mean ≈ 0.12
    lo, hi = credible_interval(alpha, beta)
    mean = alpha / (alpha + beta)
    assert lo < mean < hi


def test_credible_interval_width_shrinks_with_data():
    """More data → narrower interval."""
    lo1, hi1 = credible_interval(12,  88)    # n=100
    lo2, hi2 = credible_interval(120, 880)   # n=1000
    assert (hi2 - lo2) < (hi1 - lo1)


def test_credible_interval_90pct_narrower_than_95pct():
    lo90, hi90 = credible_interval(120, 880, level=0.90)
    lo95, hi95 = credible_interval(120, 880, level=0.95)
    assert (hi90 - lo90) < (hi95 - lo95)


# ── bayesian_ab_summary ───────────────────────────────────────────────────────

@pytest.fixture
def ab_df():
    rng = np.random.default_rng(42)
    n = 5_000
    # Same conversion array for both groups so posteriors are symmetric
    converted = list(rng.binomial(1, 0.12, n))
    return pd.DataFrame({
        "group":     ["control"] * n + ["treatment"] * n,
        "converted": converted + converted,
    })


def test_bayesian_summary_keys(ab_df):
    result = bayesian_ab_summary(ab_df)
    expected = {
        "prior", "posterior_control", "posterior_treatment",
        "mean_control", "mean_treatment",
        "ci_95_control", "ci_95_treatment",
        "prob_treatment_better", "prob_control_better",
        "loss_if_ship_treatment", "loss_if_keep_control",
    }
    assert expected <= set(result.keys())


def test_bayesian_summary_probs_sum_to_one(ab_df):
    result = bayesian_ab_summary(ab_df)
    total = result["prob_treatment_better"] + result["prob_control_better"]
    assert total == pytest.approx(1.0, abs=1e-3)


def test_bayesian_summary_means_in_range(ab_df):
    result = bayesian_ab_summary(ab_df)
    assert 0 < result["mean_control"]   < 1
    assert 0 < result["mean_treatment"] < 1


def test_bayesian_summary_ci_ordered(ab_df):
    result = bayesian_ab_summary(ab_df)
    lo_c, hi_c = result["ci_95_control"]
    lo_t, hi_t = result["ci_95_treatment"]
    assert lo_c < hi_c
    assert lo_t < hi_t


def test_bayesian_summary_null_effect_near_half(ab_df):
    """With identical rates, P(treatment better) should be near 0.5."""
    result = bayesian_ab_summary(ab_df)
    assert 0.40 <= result["prob_treatment_better"] <= 0.60


def test_bayesian_summary_informative_prior(ab_df):
    """An informative prior shifts the posterior; result should still be valid."""
    result = bayesian_ab_summary(ab_df, alpha_prior=10, beta_prior=90)
    assert 0 < result["mean_control"] < 1
    assert 0 <= result["prob_treatment_better"] <= 1
