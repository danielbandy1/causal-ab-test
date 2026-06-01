# Causal A/B Test Analysis

[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

A full causal inference analysis of a website redesign A/B test — covering classical hypothesis testing, Bayesian inference, propensity score matching, subgroup heterogeneity detection, and CUPED variance reduction. Wrapped in an interactive Streamlit dashboard.

**Dataset:** [Udacity A/B Test Results](https://www.kaggle.com/datasets/zhangluyuan/ab-testing) — 294,478 user sessions, binary conversion outcome.

---

## Quick Results

The website redesign did not improve conversion: after cleaning 290,585 sessions, treatment converted slightly worse than control, and the effect was not statistically or practically convincing. For portfolio context, this project demonstrates a full decision-quality experimentation workflow - frequentist tests, Bayesian inference, propensity score matching, subgroup checks, and CUPED - and shows how to recommend "do not ship" when the evidence does not support launch.

---

## Key Results

| Metric | Value |
|---|---|
| Raw rows | 298,306 |
| Clean rows | 290,585 |
| Control CVR | 12.04% |
| Treatment CVR | 11.88% |
| Absolute lift | −0.16 pp |
| z-test p-value | 0.1899 |
| 95% CI on lift | [−0.39 pp, +0.07 pp] |
| P(treatment > control) | ~13–28% (Bayesian, prior-dependent) |
| PSM p-value (matched) | 0.2191 |
| Subgroup heterogeneity | p = 0.83 — not significant |
| CUPED variance reduction | ~0% (hour-of-day corr ≈ 0.003) |

**Verdict: Do not ship.** Frequentist, Bayesian, and matched analyses consistently show no meaningful improvement. The null result is uniform across time-of-day subgroups.

---

## Methods

| Method | What it answers |
|---|---|
| Two-proportion z-test | Is the lift statistically significant? |
| Power analysis | Were we adequately powered to detect the observed effect? |
| Beta-Binomial Bayesian | P(treatment > control)? Expected loss of each decision? |
| Propensity Score Matching | Does time-of-day selection confound the result? |
| Subgroup analysis | Is the null result uniform, or does it hide a winning segment? |
| CUPED | Can we reduce estimator variance with a pre-experiment covariate? |

---

## Statistical Power

The observed treatment effect is -0.070 percentage points, so retrospective power is mainly a diagnostic for how small the observed difference was rather than evidence to ship. With the cleaned sample (`n_control=145,680`, `n_treatment=144,970`), the observed-effect power calculation is:

| Quantity | Value |
|---|---:|
| Observed Cohen h | -0.0025 |
| Actual power for observed effect | 10.2% |
| Required n/group for observed effect at 80% power | 2,573,321 |

For planning, a more useful minimum detectable effect is a 0.5 percentage-point absolute lift from the current 11.95% control conversion rate. Detecting that lift at 80% power and alpha=0.05 would require approximately **67,253 users per group**.

## Dashboard

The interactive Streamlit dashboard lets you adjust Bayesian priors and PSM caliper and see every result update live.

**Tabs:** Frequentist · Bayesian · Propensity Score Matching · Subgroup Analysis · CUPED · Data Quality

### Running locally

```bash
git clone https://github.com/danielbandy1/causal-ab-test
cd causal-ab-test
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_dashboard.sh        # opens at http://localhost:8502
```

---

## Project Structure

```
causal-ab-test/
├── app.py                   # Streamlit dashboard entry point
├── run_dashboard.sh         # Launch script
├── requirements.txt
├── src/
│   ├── analysis.py          # z-test, power, PSM, subgroups, CUPED
│   └── bayesian.py          # Beta-Binomial posterior, expected loss, credible intervals
├── data/
│   └── raw/ab_data.csv      # Udacity A/B test dataset
└── tests/
    ├── test_analysis.py     # 36 unit tests
    └── test_bayesian.py     # 15 unit tests
```

---

## Method Notes

### CUPED
CUPED (Kohavi et al., 2013) reduces variance in the lift estimate by regressing out a covariate *X* that is correlated with the outcome but independent of treatment:

$$\hat{Y}_{\text{cuped},i} = Y_i - \theta\,(X_i - \bar{X}), \quad \theta = \frac{\text{Cov}(Y_{\text{ctrl}},\, X_{\text{ctrl}})}{\text{Var}(X_{\text{ctrl}})}$$

θ is estimated from the control arm only to prevent treatment leakage. Here, `hour_of_day` serves as the proxy covariate (no true pre-experiment period available in this dataset). Because corr(hour, conversion) ≈ 0.003, the variance reduction is negligible — which is itself informative: time of day adds no predictive power over conversion, confirming good randomisation. In production, a prior-week conversion rate as covariate typically yields 10–50% variance reduction.

### Propensity Score Matching
Propensity scores are estimated via logistic regression on `hour` and `day_of_week`. The Rubin (2001) caliper (0.2 × SD(logit(PS))) is applied. Scores cluster tightly near 0.5, confirming that the experiment was well-randomised with respect to time — no time-of-day confounding exists.

### Bayesian Inference
Conjugate Beta-Binomial model with a symmetric Beta(α₀, β₀) prior (adjustable in the dashboard). P(treatment > control) and expected loss are estimated via 200,000 Monte Carlo samples from the posteriors.

### Data Quality
The raw dataset contains 3,894 duplicate user IDs and 3,827 rows where the landing page shown does not match the assignment group. Both are removed before analysis.

---

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
# 51 passed
```

---

## License

MIT
