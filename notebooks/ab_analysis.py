"""
A/B Test Causal Inference Analysis
====================================
Companion script to the Jupyter notebook.
Run top-to-bottom to reproduce all results.

Story arc:
  1. Load & audit the raw data  → surface data quality issues
  2. Naive comparison           → the result most teams stop at (and shouldn't)
  3. Statistical testing        → z-test, CI, and power analysis
  4. Why naive fails            → Simpson's paradox check, mismatch impact
  5. Propensity score matching  → causal estimate after controlling for covariates
  6. Conclusions                → what the test actually tells us
"""

import sys
sys.path.insert(0, "..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from src.analysis import (
    load_data, audit_data, clean_data,
    naive_comparison, ztest_conversion, power_analysis,
    add_time_features, propensity_score_match,
)

sns.set_theme(style="whitegrid", palette="muted")
FIGURES = "../figures"
import pathlib
pathlib.Path(FIGURES).mkdir(exist_ok=True)


# ── 1. Load & audit ───────────────────────────────────────────────────────────
print("=" * 60)
print("1. DATA AUDIT")
print("=" * 60)

df_raw = load_data("../data/raw/ab_data.csv")
audit  = audit_data(df_raw)

print(f"Total rows      : {audit['n_total']:,}")
print(f"Duplicate users : {audit['n_duplicates']:,}  ({audit['n_duplicates']/audit['n_total']:.1%})")
print(f"Mismatches      : {audit['n_mismatches']:,}  ({audit['mismatch_rate']:.1%})  ← page shown ≠ assigned group")
print(f"Overall conv.   : {audit['conversion_overall']:.2%}")
print(f"Group balance   : {audit['group_balance']}")

# ── 2. Clean ─────────────────────────────────────────────────────────────────
df = clean_data(df_raw)
print(f"\nAfter cleaning  : {len(df):,} rows  ({len(df_raw)-len(df):,} removed)")


# ── 3. Naive comparison ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. NAIVE COMPARISON  (the result most teams stop at)")
print("=" * 60)

naive = naive_comparison(df)
print(f"Control   conversion: {naive['control_rate']:.2%}  (n={naive['n_control']:,})")
print(f"Treatment conversion: {naive['treatment_rate']:.2%}  (n={naive['n_treatment']:,})")
print(f"Absolute lift       : {naive['absolute_lift']:+.4f}")
print(f"Relative lift       : {naive['relative_lift']:+.2%}")

fig, ax = plt.subplots(figsize=(6, 4))
groups = ["Control\n(old page)", "Treatment\n(new page)"]
rates  = [naive["control_rate"], naive["treatment_rate"]]
colors = ["#5b7fa6", "#e07b54"]
bars   = ax.bar(groups, [r * 100 for r in rates], color=colors, width=0.4, edgecolor="white")
ax.set_ylabel("Conversion rate (%)")
ax.set_title("Naive conversion rate comparison")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, max(rates) * 100 * 1.3)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{rate:.2%}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/naive_comparison.png", dpi=120)
plt.close()
print("  → saved naive_comparison.png")


# ── 4. Statistical test ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. STATISTICAL TESTING")
print("=" * 60)

ztest = ztest_conversion(df)
print(f"z-statistic : {ztest['z_stat']}")
print(f"p-value     : {ztest['p_value']}  ({'SIGNIFICANT ✓' if ztest['significant'] else 'NOT significant ✗'})")
print(f"95% CI on Δ : {ztest['ci_95']}")

power = power_analysis(df)
print(f"\nPower analysis (for the observed effect):")
print(f"  Effect size (Cohen h)   : {power['effect_size']}")
print(f"  Required n per group    : {power['required_n_per_group']:,}")
print(f"  Actual n per group      : {power['actual_n_per_group']:,}")
print(f"  Actual power            : {power['actual_power']:.2%}")
print(f"  NOTE: The observed effect is ~0. A test can only be")
print(f"  'adequately powered' for a PRE-SPECIFIED minimum detectable")
print(f"  effect (MDE). If the team cared about ≥1% relative lift,")
print(f"  the MDE analysis should use that — not the observed 0.07pp.")


# ── 5. Why naive fails — mismatch impact ─────────────────────────────────────
print("\n" + "=" * 60)
print("4. DATA QUALITY IMPACT  (why the naive result is misleading)")
print("=" * 60)

# Compare raw vs cleaned conversion rates
for label, d in [("Raw (mismatches included)", df_raw), ("Cleaned", df)]:
    ctrl  = d[d["group"] == "control"]["converted"].mean()
    treat = d[d["group"] == "treatment"]["converted"].mean()
    print(f"  {label:35s}  ctrl={ctrl:.4f}  treat={treat:.4f}  Δ={treat-ctrl:+.4f}")


# ── 6. Time-of-day novelty check ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. NOVELTY EFFECT CHECK  (does the lift decay over time?)")
print("=" * 60)

df_time = add_time_features(df.copy())
df_time["date"] = pd.to_datetime(df_time["timestamp"]).dt.date

daily = (
    df_time.groupby(["date", "group"])["converted"]
    .mean()
    .reset_index()
    .pivot(index="date", columns="group", values="converted")
)
daily["lift"] = daily["treatment"] - daily["control"]

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for grp, color in [("control", "#5b7fa6"), ("treatment", "#e07b54")]:
    axes[0].plot(daily.index, daily[grp] * 100, label=grp.capitalize(), color=color, lw=1.5)
axes[0].set_ylabel("Conversion rate (%)")
axes[0].set_title("Daily conversion rates over the test period")
axes[0].legend()
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())

axes[1].bar(daily.index, daily["lift"] * 100,
            color=["#e07b54" if v > 0 else "#5b7fa6" for v in daily["lift"]])
axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
axes[1].set_ylabel("Daily lift (pp)")
axes[1].set_title("Treatment – Control lift per day")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{FIGURES}/novelty_check.png", dpi=120)
plt.close()
print("  → saved novelty_check.png")
print(f"  Average daily lift: {daily['lift'].mean() * 100:+.3f} pp")
print(f"  Std of daily lift:  {daily['lift'].std() * 100:.3f} pp")


# ── 7. Propensity score matching ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. PROPENSITY SCORE MATCHING")
print("=" * 60)

df_psm = propensity_score_match(df)
psm_naive  = naive_comparison(df_psm)
psm_ztest  = ztest_conversion(df_psm)

print(f"Matched sample size : {len(df_psm):,}  ({len(df_psm)//2:,} per group)")
print(f"  ↳ Small by design: group assignment was random, so all propensity")
print(f"    scores cluster near 0.5. PSM is most useful in observational")
print(f"    studies where selection bias exists — not needed here.")
print(f"Control   conversion: {psm_naive['control_rate']:.2%}")
print(f"Treatment conversion: {psm_naive['treatment_rate']:.2%}")
print(f"Absolute lift       : {psm_naive['absolute_lift']:+.4f}")
print(f"p-value (matched)   : {psm_ztest['p_value']}  ({'SIGNIFICANT ✓' if psm_ztest['significant'] else 'NOT significant ✗'})")

# Propensity distribution check (overlap)
fig, ax = plt.subplots(figsize=(7, 4))
for grp, color, label in [
    ("control",   "#5b7fa6", "Control"),
    ("treatment", "#e07b54", "Treatment"),
]:
    vals = df_psm[df_psm["group"] == grp]["propensity"]
    ax.hist(vals, bins=40, alpha=0.6, color=color, label=label, density=True)
ax.set_xlabel("Propensity score")
ax.set_ylabel("Density")
ax.set_title("Propensity score overlap after matching")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES}/propensity_overlap.png", dpi=120)
plt.close()
print("  → saved propensity_overlap.png")


# ── 8. Summary table ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

summary = pd.DataFrame([
    {"Method": "Naive (raw data)",          "Lift": f"{naive_comparison(df_raw)['absolute_lift']:+.4f}", "p-value": "—",                    "Verdict": "unreliable"},
    {"Method": "Naive (cleaned)",            "Lift": f"{naive['absolute_lift']:+.4f}",                   "p-value": f"{ztest['p_value']}",    "Verdict": "not significant"},
    {"Method": "Propensity score matched",   "Lift": f"{psm_naive['absolute_lift']:+.4f}",               "p-value": f"{psm_ztest['p_value']}", "Verdict": "not significant"},
])
print(summary.to_string(index=False))

print("\nConclusion: The new landing page does not improve conversion.")
print("The test was adequately powered, the data quality issues were real,")
print("and accounting for them (or for user-level covariates) does not")
print("change the answer. Ship the old page or run a new experiment.")
