# Causal A/B Test Analysis

This project analyzes a website redesign experiment using both classical A/B testing and causal inference methods. It is designed as a realistic product analytics case study: audit the data, estimate the treatment effect, quantify uncertainty, check whether the effect varies by subgroup, and turn the result into a product recommendation.

## Executive Summary

A new landing page was tested against an existing page on roughly 298,000 user observations. A naive comparison is not enough because the raw data contains duplicate users and page/group mismatches. After cleaning, the treatment does not improve conversion.

| Result | Value |
| --- | ---: |
| Raw rows | 298,306 |
| Clean rows | 290,650 |
| Control conversion | 11.95% |
| Treatment conversion | 11.87% |
| Absolute lift | -0.07 percentage points |
| Two-proportion z-test p-value | 0.5575 |
| 95% CI on lift | [-0.31pp, +0.17pp] |
| Bayesian P(treatment better) | 27.97% |
| Heterogeneity test p-value | 0.1482 |

Recommendation: do not ship the redesign. The evidence consistently favors keeping the control or redesigning the experiment around a stronger product hypothesis.

## What This Project Demonstrates

| Skill | Implementation |
| --- | --- |
| Data quality auditing | Detect duplicate users and treatment/page mismatches |
| Frequentist testing | Two-proportion z-test and confidence interval |
| Power analysis | Observed-effect power and required sample sizing |
| Causal inference | Propensity score matching with Rubin-style caliper |
| Bayesian testing | Beta-Binomial posterior, probability of superiority, expected loss |
| Heterogeneity analysis | Subgroup lift estimates and logistic interaction test |
| Communication | Business recommendation framed for product decision-making |
| Software engineering | Modular `src/`, generated notebook, pytest suite |

## Data Quality

The raw dataset has two issues that must be addressed before estimating causal effects:

- Duplicate users: 3,828
- Page/group mismatches: 3,878
- Mismatch rate: 1.3%

The cleaning step removes duplicate users and rows where treatment assignment does not match the landing page shown:

- Treatment group shown old page
- Control group shown new page

This creates a cleaner experiment table with 290,650 rows.

## Analysis Workflow

### 1. Naive Comparison

`naive_comparison()` computes conversion rates by group. On cleaned data:

| Group | Conversion | Users |
| --- | ---: | ---: |
| Control | 11.95% | 145,680 |
| Treatment | 11.87% | 144,970 |

The treatment underperforms by about 0.07 percentage points.

### 2. Frequentist Hypothesis Test

`ztest_conversion()` performs a two-proportion z-test:

- Null: treatment conversion equals control conversion
- Alternative: treatment conversion differs from control conversion
- p-value: 0.5575
- 95% CI: -0.31pp to +0.17pp

The interval includes zero and the upper bound is commercially small.

### 3. Power Analysis

`power_analysis()` estimates the sample required to detect the observed effect. Because the observed effect is tiny, the required sample size for that exact effect is very large. This is an important distinction: a null result can mean either the experiment is underpowered for a tiny effect or the tested design simply does not move the business metric enough to matter.

### 4. Propensity Score Matching

Although this is an experiment, `propensity_score_match()` demonstrates causal adjustment using time-based covariates:

- Adds hour-of-day and day-of-week features.
- Estimates treatment propensity with logistic regression.
- Matches treatment and control users by nearest propensity score within a caliper.

The matched sample is intentionally small because a well-randomized experiment has propensities clustered near 0.5; overlap diagnostics are part of the lesson.

### 5. Bayesian A/B Testing

`src/bayesian.py` models conversion rates with Beta-Binomial posteriors:

- Prior: Beta(1, 1)
- Posterior control: Beta(17403, 128279)
- Posterior treatment: Beta(17216, 127756)
- Posterior mean control: 0.119459
- Posterior mean treatment: 0.118754
- P(treatment better): 0.2797
- P(control better): 0.7204

The Bayesian framing reinforces the frequentist conclusion and adds decision-focused expected-loss metrics.

### 6. Heterogeneity and Subgroups

`subgroup_analysis()` checks whether the aggregate null result hides a winning segment:

- Hour buckets: Night, Morning, Afternoon, Evening
- Day type: Weekday, Weekend
- Bonferroni correction for multiple tests

No subgroup is statistically significant after correction. The logistic interaction test also does not find significant heterogeneity (`p = 0.1482`).

## Figures

Generated figures:

```text
figures/naive_comparison.png
figures/novelty_check.png
figures/propensity_overlap.png
```

These support the story visually: raw conversion comparison, time-pattern/novelty checks, and propensity overlap.

## Repository Structure

```text
causal-ab-test/
├── data/raw/ab_data.csv
├── figures/
├── notebooks/
│   ├── ab_analysis.ipynb
│   └── ab_analysis.py
├── src/
│   ├── analysis.py       # cleaning, z-test, power, matching, subgroups
│   └── bayesian.py       # Beta-Binomial Bayesian analysis
├── tests/
│   ├── test_analysis.py
│   └── test_bayesian.py
├── build_notebook.py
├── download_data.py
├── generate_data.py
└── requirements.txt
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Get data:

```bash
python3 download_data.py
```

Launch the notebook:

```bash
jupyter notebook notebooks/ab_analysis.ipynb
```

Run tests:

```bash
python3 -m pytest tests/ -v
```

## Business Recommendation

Do not ship the new landing page.

The treatment has lower observed conversion, the z-test is null, the credible posterior probability of treatment superiority is below 50%, expected loss is higher if treatment ships, and subgroup analysis does not reveal a hidden winning segment.

Recommended next steps:

- Investigate downstream funnel drop-off before redesigning the landing page again.
- Use qualitative research to identify a specific conversion blocker.
- Define a stronger hypothesis tied to a measurable mechanism.
- Pre-register the minimum detectable effect and stopping rule before the next experiment.

## Portfolio Highlights

This project shows practical causal analysis judgment: it does not just run a p-value. It audits assignment integrity, separates raw and cleaned estimates, compares frequentist and Bayesian conclusions, checks heterogeneity, and translates uncertainty into a product decision.

