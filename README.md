# Causal Inference in A/B Testing

> Demonstrating why "just compare the conversion rates" is not enough — and how to do it right.

A complete walkthrough of an A/B test analysis using a 298,000-user website redesign
experiment. The project deliberately shows both the naive approach and the causal
approach, explaining at each step why the naive result can mislead.

---

## What This Project Demonstrates

| Skill | Implementation |
|---|---|
| Data quality auditing | Detecting page/group mismatches and duplicate user IDs before analysis |
| Hypothesis testing | Two-proportion z-test, 95% confidence intervals |
| Experimental design | Pre-specified MDE vs. post-hoc power analysis (and why they differ) |
| Novelty effect detection | Time-series decomposition of daily lift |
| Causal inference | Propensity score matching (PSM) with Rubin caliper |
| Communication | Narrative notebook structured for non-technical stakeholders |

---

## Key Findings

1. **Data quality trap**: 1.3% of users were assigned to the wrong page — a real data
   pipeline bug that contaminates the naive lift estimate.

2. **No significant effect**: After cleaning, p = 0.56. The new landing page does not
   improve conversion. The 95% CI on the lift is (−0.31pp, +0.17pp).

3. **Power analysis nuance**: The observed effect is ~0.07pp — effectively noise. A
   test can only be "adequately powered" relative to a pre-specified MDE. At our
   sample size (~145k per group), the test had >80% power to detect a ≥1% relative
   lift, so the null result is trustworthy.

4. **PSM confirms the result**: Matching on time-of-day covariates yields the same
   conclusion. The small matched sample is expected in a randomised experiment —
   PSM is most valuable in observational studies where selection bias exists.

**Recommendation**: Do not ship the new page. Run a new experiment with a different
design hypothesis.

---

## Project Structure

```
causal-ab-test/
├── data/
│   └── raw/
│       └── ab_data.csv          # Generated locally (see below)
├── figures/                     # Saved PNGs for README / slides
│   ├── naive_comparison.png
│   ├── novelty_check.png
│   └── propensity_overlap.png
├── notebooks/
│   └── ab_analysis.ipynb        # Main analysis notebook
├── src/
│   └── analysis.py              # Reusable analysis functions
├── build_notebook.py            # Regenerates the .ipynb file
├── generate_data.py             # Synthetic dataset generator
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

This tries several public mirrors for the original Udacity A/B Test Results dataset.
If all mirrors are unreachable, it generates a synthetic dataset with identical
statistical properties (same conversion rates, same data quality issues).

### 3. Launch the notebook

```bash
jupyter notebook notebooks/ab_analysis.ipynb
```

Or run the analysis as a plain Python script:

```bash
cd notebooks && python3 ab_analysis.py
```

---

## Dataset

Based on the **Udacity A/B Test Results** dataset (used in Udacity's A/B Testing
course). The synthetic version reproduces:

- ~298,000 rows, ~50/50 control/treatment split
- Control conversion rate: 11.95%
- Treatment conversion rate: 11.87%
- 1.3% page/group mismatches (data quality issue)
- 1.3% duplicate user IDs
- Timestamps spanning Jan–Mar 2017 with realistic hour-of-day traffic weights

---

## Results at a Glance

| Method | n per group | Lift (pp) | p-value | Verdict |
|---|---|---|---|---|
| Naive (dirty data) | ~149k | −0.0009 | — | ⚠️ unreliable |
| Naive (cleaned) | ~145k | −0.0007 | 0.558 | ✗ not significant |
| Propensity matched | 168 | −0.0298 | 0.414 | ✗ not significant |
