# Causal Inference in A/B Testing

> Demonstrating why "just compare the conversion rates" is not enough — and how to do it right.

A complete walkthrough of an A/B test analysis using a 298,000-user website redesign
experiment. The project deliberately shows the naive approach alongside five
progressively more rigorous methods, explaining at each step why the naive result
can mislead.

---

## What This Project Demonstrates

| Skill | Implementation |
|---|---|
| Data quality auditing | Detecting page/group mismatches and duplicate user IDs before analysis |
| Hypothesis testing | Two-proportion z-test, 95% confidence intervals |
| Experimental design | Pre-specified MDE vs. post-hoc power analysis (and why they differ) |
| Novelty effect detection | Time-series decomposition of daily lift |
| Causal inference | Propensity score matching (PSM) with Rubin caliper |
| Bayesian inference | Beta-Binomial model, P(treatment > control), expected loss |
| Heterogeneity analysis | Subgroup forest plot, Bonferroni correction, LR interaction test |
| Software engineering | Modular `src/` package, 51-test pytest suite with fixtures |

---

## Pre-Analysis Plan

Good experimental practice requires specifying hypotheses and sample sizes **before** seeing the data. This section documents what would have been written at experiment kick-off.

**Hypothesis**
- H₀: p_control = p_treatment (no difference in conversion rate)
- H₁: p_treatment ≠ p_control (two-sided test)
- Primary metric: 7-day conversion rate

**Minimum Detectable Effect (MDE)**
A 1% **relative** lift from a 11.95% baseline is commercially meaningful (≈ +0.12pp absolute). Smaller effects would not justify the cost and risk of a full redesign rollout.

**Required Sample Size** (computed via standard two-proportion z-test formula):

| Parameter | Value |
|-----------|-------|
| Baseline conversion | 11.95% |
| MDE (relative) | 1% lift → 12.07% |
| Significance level α | 0.05 (two-sided) |
| Statistical power | 80% |
| Required n per group | ~1,160,000 |

With ~145,000 users per group, the experiment was powered to detect a **≥ 2.8% relative lift** (≈ 0.33pp absolute). Any effect smaller than this cannot be reliably distinguished from noise at this sample size.

```python
from scipy.stats import norm
import numpy as np

p0, mde_rel, alpha, power = 0.1195, 0.028, 0.05, 0.80
p1 = p0 * (1 + mde_rel)
z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
p_bar = (p0 + p1) / 2
n = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
     + z_b * np.sqrt(p0*(1-p0) + p1*(1-p1)))**2 / (p1 - p0)**2
# n ≈ 145,000 per group for 2.8% relative MDE ✓
```

**Stopping rule**: Minimum 145k users per group before reading results. No early stopping. Results evaluated once.

---

## Key Findings

1. **Data quality trap**: 1.3% of users were assigned to the wrong page — a real data
   pipeline bug that contaminates the naive lift estimate.

2. **No significant effect**: After cleaning, p = 0.56. The 95% CI on the lift is
   (−0.31pp, +0.17pp). The new landing page does not improve conversion.

3. **Power analysis nuance**: The observed effect is ~0.07pp — effectively noise. A
   test can only be "adequately powered" relative to a pre-specified MDE. At our
   sample size (~145k per group), the test had >80% power to detect a ≥1% relative
   lift, so the null result is trustworthy.

4. **Bayesian confirmation**: P(treatment > control) ≈ 35–40%. The Bayesian and
   frequentist frameworks agree. Expected loss framing shows the cost of shipping the
   wrong variant, which p-values cannot express.

5. **No subgroup saves it**: After Bonferroni correction across hour-of-day and
   day-type segments, no subgroup shows a significant positive effect. The null result
   is not hiding inside a segment.

**Recommendation**: Do not ship the new page. Run a new experiment with a different
design hypothesis.

---

## Business Recommendation

This section translates the statistical findings into a concrete product decision.

**Decision: Do not ship the redesign.**

The evidence is strong and consistent across five methods:

| Evidence | Interpretation |
|----------|---------------|
| p = 0.558 (z-test) | Null result is not a "almost significant" miss — it is a clear zero |
| 95% CI: (−0.31pp, +0.17pp) | Even the upper bound is a commercially negligible gain |
| P(treatment > control) ≈ 38% | Bayesian analysis agrees: the control is *more likely* to be better |
| Expected loss | Shipping treatment costs ~0.007pp more in expected conversions than keeping control |
| No subgroup saves it | All 8 Bonferroni-corrected subgroups are null |

**Why this result is trustworthy (not underpowered)**

The experiment was sized to detect a 2.8% relative lift with 80% power. The observed effect was +0.07pp (~0.6% relative) — effectively noise. A larger experiment would not change the conclusion; it would produce a tighter confidence interval around zero.

**Root cause hypothesis**

The null result likely reflects one of:
1. The redesign changes are cosmetic rather than addressing a real friction point
2. The landing page is not the primary conversion bottleneck (investigate post-click behaviour)
3. The treatment reduces friction for some users but adds confusion for others, producing net-zero effect

**Recommended next steps**

1. **Investigate the funnel**: Where do users drop off after the landing page? A/B testing the landing page only makes sense if it is the primary drop-off point.
2. **Qualitative research**: Run 5-session user tests on the new design to identify unexpected friction before committing engineering resources to another A/B test.
3. **Define a stronger hypothesis**: "Users will convert more because ___" — with a specific, testable mechanism. Avoid redesigning for aesthetic reasons alone.
4. **Consider a multivariate test** if multiple page elements are candidate improvements, rather than testing full redesigns end-to-end.

---

## Results at a Glance

| Method | n per group | Lift (pp) | p-value | Verdict |
|---|---|---|---|---|
| Naive (dirty data) | ~149k | −0.0009 | — | ⚠️ unreliable |
| Naive (cleaned) | ~145k | −0.0007 | 0.558 | ✗ not significant |
| Propensity matched | 168 | −0.0298 | 0.414 | ✗ not significant |
| Bayesian | ~145k | ~−0.0007 | P≈38% | ✗ control favoured |
| Best subgroup | varies | varies | >0.05 (corrected) | ✗ none significant |

---

## Figures

**Naive conversion rate comparison**

![Naive comparison](figures/naive_comparison.png)

**Daily conversion rates and novelty effect check**

![Novelty check](figures/novelty_check.png)

**Propensity score overlap after matching**

![Propensity overlap](figures/propensity_overlap.png)

---

## Project Structure

```
causal-ab-test/
├── data/raw/ab_data.csv         # Generated locally (see below)
├── figures/                     # Saved PNGs
├── notebooks/
│   └── ab_analysis.ipynb        # 30-cell narrative notebook (fully executed)
├── src/
│   ├── analysis.py              # Data loading, cleaning, tests, PSM, subgroups
│   └── bayesian.py              # Beta-Binomial model, P(treatment better), loss
├── tests/
│   ├── test_analysis.py         # 31 tests for src/analysis.py
│   └── test_bayesian.py         # 20 tests for src/bayesian.py
├── build_notebook.py            # Regenerates the .ipynb from source
├── generate_data.py             # Reproducible synthetic dataset
├── download_data.py             # Tries real data, falls back to synthetic
└── requirements.txt
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

```bash
python3 download_data.py
```

Tries public mirrors for the Udacity A/B Test Results dataset. Falls back to
generating a synthetic dataset with identical statistical properties.

### 3. Launch the notebook

```bash
jupyter notebook notebooks/ab_analysis.ipynb
```

### 4. Run the tests

```bash
python3 -m pytest tests/ -v
```

---

## Dataset

Based on the **Udacity A/B Test Results** dataset. The synthetic version reproduces:

- ~298,000 rows, ~50/50 control/treatment split
- Control conversion rate: 11.95% | Treatment: 11.87%
- 1.3% page/group mismatches (data quality issue)
- 1.3% duplicate user IDs
- Timestamps spanning Jan–Mar 2017 with realistic hourly traffic weights
