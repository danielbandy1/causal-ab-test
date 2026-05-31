"""
Causal A/B Test Dashboard
Interactive analysis of a website redesign experiment using frequentist,
Bayesian, and causal inference methods.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats

# Resolve src/ regardless of working directory
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import (
    load_data, audit_data, clean_data,
    naive_comparison, ztest_conversion,
    power_analysis, propensity_score_match,
    subgroup_analysis, heterogeneity_test,
    cuped_adjustment,
)
from bayesian import bayesian_ab_summary, credible_interval

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Causal A/B Test · Daniel Bandy",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .verdict-box {
    padding: 18px 22px; border-radius: 12px;
    font-size: 1.05rem; line-height: 1.6;
  }
  .verdict-no  { background: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b; }
  .verdict-yes { background: #f0fdf4; border-left: 5px solid #22c55e; color: #166534; }
  .kpi-label   { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; }
  .kpi-value   { font-size: 2rem; font-weight: 800; line-height: 1.1; }
  .kpi-sub     { font-size: 0.82rem; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────

DATA_PATH = ROOT / "data" / "raw" / "ab_data.csv"

@st.cache_data
def get_data():
    raw  = load_data(str(DATA_PATH))
    info = audit_data(raw)
    clean = clean_data(raw)
    return raw, info, clean

@st.cache_data
def get_freq(clean):
    naive = naive_comparison(clean)
    ztest = ztest_conversion(clean)
    power = power_analysis(clean)
    return naive, ztest, power

@st.cache_data
def get_bayes(clean, a0, b0):
    return bayesian_ab_summary(clean, alpha_prior=a0, beta_prior=b0)

@st.cache_data
def get_psm(clean, caliper):
    cal = caliper if caliper > 0 else None
    matched = propensity_score_match(clean, caliper=cal)
    naive_m = naive_comparison(matched)
    ztest_m = ztest_conversion(matched)
    return matched, naive_m, ztest_m

@st.cache_data
def get_subgroups(clean):
    subs = subgroup_analysis(clean)
    htest = heterogeneity_test(clean)
    return subs, htest

@st.cache_data
def get_cuped(clean):
    return cuped_adjustment(clean)

raw, info, clean = get_data()

# ── Sidebar controls ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧪 Controls")
    st.caption("Adjust parameters to re-run the analysis.")

    st.subheader("Bayesian prior")
    alpha_prior = st.slider("α₀ (prior successes)", 0.5, 10.0, 1.0, 0.5)
    beta_prior  = st.slider("β₀ (prior failures)",  0.5, 10.0, 1.0, 0.5)

    st.subheader("PSM caliper")
    caliper = st.slider(
        "Caliper (0 = Rubin rule)", 0.0, 0.10, 0.0, 0.005,
        format="%.3f",
        help="Max allowed propensity score difference for a match. 0 uses 0.2×SD(logit(PS)).",
    )

    st.divider()
    st.caption("**Dataset:** Udacity A/B Test Results — website redesign study")
    st.caption("**Methods:** z-test · Beta-Binomial · PSM · LR interaction test")
    st.caption("**Author:** Daniel Bandy · [GitHub](https://github.com/danielbandy1)")

# ── Load analysis ──────────────────────────────────────────────────────────────

naive, ztest, power = get_freq(clean)
bayes = get_bayes(clean, alpha_prior, beta_prior)
matched, naive_m, ztest_m = get_psm(clean, caliper)
subs, htest = get_subgroups(clean)
cuped = get_cuped(clean)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("Causal A/B Test Analysis")
st.caption(
    "A website redesign experiment analysed with classical hypothesis testing, "
    "Bayesian inference, and propensity-score matching. "
    "Adjust parameters in the sidebar to see results update live."
)

ship = bayes["prob_treatment_better"] > 0.95 and ztest["significant"]
verdict_cls  = "verdict-yes" if ship else "verdict-no"
verdict_text = (
    "✅ **Ship the redesign.** Both frequentist and Bayesian tests show a significant positive effect."
    if ship else
    "🚫 **Do not ship.** The redesign shows no meaningful improvement in conversion. "
    "Evidence consistently favours keeping the control or redesigning the hypothesis."
)
st.markdown(
    f'<div class="verdict-box {verdict_cls}">{verdict_text}</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Top KPIs ───────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>',
        unsafe_allow_html=True,
    )

kpi(k1, "Clean rows",    f"{len(clean):,}",
    f"{info['n_duplicates']:,} dupes + {info['n_mismatches']:,} mismatches removed")
kpi(k2, "Control CVR",   f"{naive['control_rate']:.2%}",   f"n = {naive['n_control']:,}")
kpi(k3, "Treatment CVR", f"{naive['treatment_rate']:.2%}", f"n = {naive['n_treatment']:,}")
kpi(k4, "Abs. lift",     f"{naive['absolute_lift']:+.3%}",
    f"CI [{ztest['ci_95'][0]:+.3%}, {ztest['ci_95'][1]:+.3%}]")
kpi(k5, "P(treat > ctrl)", f"{bayes['prob_treatment_better']:.1%}", "Bayesian")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_freq, tab_bayes, tab_psm, tab_subgroup, tab_cuped, tab_quality = st.tabs([
    "📊 Frequentist Test",
    "🎲 Bayesian Analysis",
    "🎯 Propensity Score Matching",
    "🔍 Subgroup Analysis",
    "⚡ CUPED",
    "🗂 Data Quality",
])

# ─── TAB 1: Frequentist ───────────────────────────────────────────────────────

with tab_freq:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.subheader("Conversion rate comparison")

        groups = ["Control", "Treatment"]
        rates  = [naive["control_rate"], naive["treatment_rate"]]
        ci_lo  = [
            rates[0] - 1.96 * np.sqrt(rates[0]*(1-rates[0])/naive["n_control"]),
            rates[1] - 1.96 * np.sqrt(rates[1]*(1-rates[1])/naive["n_treatment"]),
        ]
        ci_hi  = [
            rates[0] + 1.96 * np.sqrt(rates[0]*(1-rates[0])/naive["n_control"]),
            rates[1] + 1.96 * np.sqrt(rates[1]*(1-rates[1])/naive["n_treatment"]),
        ]
        colors = ["#3b82f6", "#f97316"]

        fig = go.Figure()
        for i, (g, r, lo, hi, c) in enumerate(zip(groups, rates, ci_lo, ci_hi, colors)):
            fig.add_trace(go.Bar(
                name=g, x=[g], y=[r],
                marker_color=c, opacity=0.85,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[hi - r],
                    arrayminus=[r - lo],
                    color="#374151",
                    thickness=2, width=8,
                ),
                text=[f"{r:.2%}"], textposition="outside",
                hovertemplate=f"<b>{g}</b><br>CVR: {r:.4%}<br>95% CI: [{lo:.4%}, {hi:.4%}]<extra></extra>",
            ))

        fig.update_layout(
            showlegend=False, height=320,
            yaxis=dict(tickformat=".1%", title="Conversion rate"),
            margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Test results")

        m1, m2 = st.columns(2)
        m1.metric("z-statistic",    f"{ztest['z_stat']:.4f}")
        m2.metric("p-value",        f"{ztest['p_value']:.4f}",
                  delta="significant" if ztest["significant"] else "not significant",
                  delta_color="normal" if ztest["significant"] else "inverse")
        m1.metric("95% CI low",     f"{ztest['ci_95'][0]:+.4%}")
        m2.metric("95% CI high",    f"{ztest['ci_95'][1]:+.4%}")

        st.divider()
        st.subheader("Power analysis")
        pm1, pm2 = st.columns(2)
        pm1.metric("Effect size (Cohen h)", f"{power['effect_size']:.4f}")
        pm2.metric("Actual power",          f"{power['actual_power']:.2%}")
        pm1.metric("Required n / group",    f"{power['required_n_per_group']:,}")
        pm2.metric("Actual n / group",      f"{power['actual_n_per_group']:,}")

        powered = power["adequately_powered"]
        st.info(
            f"✅ Experiment is adequately powered at {power['actual_power']:.0%}."
            if powered else
            f"⚠️ Under-powered at {power['actual_power']:.0%} — "
            f"would need {power['required_n_per_group']:,} per group to reach 80%."
        )

# ─── TAB 2: Bayesian ──────────────────────────────────────────────────────────

with tab_bayes:
    col_l, col_r = st.columns([1.2, 1], gap="large")

    with col_l:
        st.subheader("Posterior distributions")

        a_c, b_c   = bayes["posterior_control"]
        a_t, b_t   = bayes["posterior_treatment"]
        x = np.linspace(0.115, 0.125, 1000)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, a_c, b_c),
            name="Control", fill="tozeroy",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.15)",
            hovertemplate="θ = %{x:.4%}<br>density = %{y:.1f}<extra>Control</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, a_t, b_t),
            name="Treatment", fill="tozeroy",
            line=dict(color="#f97316", width=2),
            fillcolor="rgba(249,115,22,0.15)",
            hovertemplate="θ = %{x:.4%}<br>density = %{y:.1f}<extra>Treatment</extra>",
        ))

        for (lo, hi), label, color in [
            (bayes["ci_95_control"],   "Control 95% CI",   "#3b82f6"),
            (bayes["ci_95_treatment"], "Treatment 95% CI", "#f97316"),
        ]:
            fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.08, line_width=0)

        fig.update_layout(
            height=340,
            xaxis=dict(tickformat=".2%", title="Conversion rate θ"),
            yaxis=dict(title="Posterior density"),
            legend=dict(orientation="h", y=1.05),
            margin=dict(t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Prior: Beta({alpha_prior}, {beta_prior})  ·  "
            f"Posterior control: Beta({a_c:.0f}, {b_c:.0f})  ·  "
            f"Posterior treatment: Beta({a_t:.0f}, {b_t:.0f})"
        )

    with col_r:
        st.subheader("Decision metrics")

        p_better = bayes["prob_treatment_better"]
        color = "#22c55e" if p_better > 0.95 else "#f97316" if p_better > 0.5 else "#ef4444"

        st.markdown(
            f"<div style='font-size:1rem;color:#6b7280;margin-bottom:4px;'>P(treatment > control)</div>"
            f"<div style='font-size:2.8rem;font-weight:800;color:{color};line-height:1;'>"
            f"{p_better:.1%}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Estimated via 200,000 Monte Carlo samples from posteriors.")
        st.divider()

        m1, m2 = st.columns(2)
        m1.metric("Control posterior mean",   f"{bayes['mean_control']:.4%}")
        m2.metric("Treatment posterior mean",  f"{bayes['mean_treatment']:.4%}")
        m1.metric("Control 95% CI",
                  f"[{bayes['ci_95_control'][0]:.4%}, {bayes['ci_95_control'][1]:.4%}]")
        m2.metric("Treatment 95% CI",
                  f"[{bayes['ci_95_treatment'][0]:.4%}, {bayes['ci_95_treatment'][1]:.4%}]")

        st.divider()
        st.subheader("Expected loss (regret)")
        el1, el2 = st.columns(2)
        el1.metric("If we ship treatment",
                   f"{bayes['loss_if_ship_treatment']:.5%}",
                   help="Risk of choosing treatment when control is actually better")
        el2.metric("If we keep control",
                   f"{bayes['loss_if_keep_control']:.5%}",
                   help="Opportunity cost of not shipping when treatment is actually better")

        if bayes["loss_if_ship_treatment"] < bayes["loss_if_keep_control"]:
            st.success("Loss-minimising decision: **ship** the treatment.")
        else:
            st.error("Loss-minimising decision: **keep** the control.")

# ─── TAB 3: PSM ───────────────────────────────────────────────────────────────

with tab_psm:
    col_l, col_r = st.columns([1.2, 1], gap="large")

    with col_l:
        st.subheader("Propensity score distribution")

        from analysis import add_time_features, propensity_score_match
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        @st.cache_data
        def get_ps(clean_df):
            df = add_time_features(clean_df.copy())
            X  = StandardScaler().fit_transform(df[["hour", "day_of_week"]].values)
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X, (df["group"] == "treatment").astype(int).values)
            df["propensity"] = lr.predict_proba(X)[:, 1]
            return df

        df_ps = get_ps(clean)

        fig = go.Figure()
        for grp, color, name in [("control","#3b82f6","Control"), ("treatment","#f97316","Treatment")]:
            ps_vals = df_ps[df_ps["group"] == grp]["propensity"].values
            fig.add_trace(go.Histogram(
                x=ps_vals, name=name,
                nbinsx=50, opacity=0.65,
                marker_color=color,
                hovertemplate=f"PS bin: %{{x:.3f}}<br>Count: %{{y}}<extra>{name}</extra>",
            ))

        fig.update_layout(
            barmode="overlay", height=300,
            xaxis=dict(title="Propensity score", range=[0, 1]),
            yaxis=dict(title="Count"),
            legend=dict(orientation="h", y=1.05),
            margin=dict(t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Propensity = P(treatment | hour, day_of_week) estimated via logistic regression. "
            "Scores cluster near 0.5 confirming this was a well-randomised experiment — "
            "time of day adds almost no predictive power."
        )

    with col_r:
        st.subheader("Matched sample results")

        n_matched = len(matched) // 2
        m1, m2, m3 = st.columns(3)
        m1.metric("Matched pairs", f"{n_matched:,}")
        m2.metric("Match rate", f"{n_matched / naive['n_control']:.1%}")
        m3.metric("p-value (matched)", f"{ztest_m['p_value']:.4f}",
                  delta="significant" if ztest_m["significant"] else "not significant",
                  delta_color="normal" if ztest_m["significant"] else "inverse")

        st.divider()

        compare_data = {
            "Sample":        ["Unmatched", "Matched"],
            "Control CVR":   [f"{naive['control_rate']:.4%}",   f"{naive_m['control_rate']:.4%}"],
            "Treatment CVR": [f"{naive['treatment_rate']:.4%}", f"{naive_m['treatment_rate']:.4%}"],
            "Abs. lift":     [f"{naive['absolute_lift']:+.4%}", f"{naive_m['absolute_lift']:+.4%}"],
            "p-value":       [f"{ztest['p_value']:.4f}",        f"{ztest_m['p_value']:.4f}"],
        }
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

        st.info(
            "PSM result is consistent with the naive comparison: no significant effect. "
            "The experiment was well-randomised, so matching changes little — "
            "itself a useful finding (no hidden confounders from time-of-day selection)."
        )

# ─── TAB 4: Subgroups ─────────────────────────────────────────────────────────

with tab_subgroup:
    col_l, col_r = st.columns([1.4, 1], gap="large")

    with col_l:
        st.subheader("Lift by subgroup (forest plot)")

        subs_sorted = subs.sort_values(["subgroup_type", "lift"])
        labels = subs_sorted["subgroup"].tolist()
        lifts  = subs_sorted["lift"].tolist()
        ci_lo  = subs_sorted["ci_low"].tolist()
        ci_hi  = subs_sorted["ci_high"].tolist()
        sig    = subs_sorted["significant"].tolist()

        colors = ["#22c55e" if s else "#94a3b8" for s in sig]

        fig = go.Figure()
        fig.add_vline(x=0, line_dash="dash", line_color="#9ca3af", line_width=1)
        fig.add_trace(go.Scatter(
            x=lifts, y=labels,
            mode="markers",
            marker=dict(color=colors, size=10, symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[hi - l for l, hi in zip(lifts, ci_hi)],
                arrayminus=[l - lo for l, lo in zip(lifts, ci_lo)],
                color="#6b7280", thickness=2, width=6,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Lift: %{x:+.3%}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=340,
            xaxis=dict(tickformat="+.2%", title="Absolute lift (treatment − control)"),
            yaxis=dict(title=""),
            margin=dict(t=20, b=20, l=140),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Subgroup table")
        display_cols = ["subgroup", "n_control", "n_treatment",
                        "control_rate", "treatment_rate", "lift", "p_bonferroni", "significant"]
        st.dataframe(
            subs[display_cols].rename(columns={
                "subgroup": "Subgroup", "n_control": "N ctrl",
                "n_treatment": "N treat", "control_rate": "Ctrl CVR",
                "treatment_rate": "Treat CVR", "lift": "Lift",
                "p_bonferroni": "p (Bonf.)", "significant": "Sig?",
            }).style.format({
                "Ctrl CVR": "{:.2%}", "Treat CVR": "{:.2%}",
                "Lift": "{:+.3%}", "p (Bonf.)": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.divider()
        st.subheader("Interaction test (heterogeneity)")
        h1, h2 = st.columns(2)
        h1.metric("LR stat",  f"{htest['lr_stat']:.4f}")
        h2.metric("p-value",  f"{htest['p_value']:.4f}",
                  delta="heterogeneous" if htest["heterogeneity"] else "homogeneous",
                  delta_color="normal" if htest["heterogeneity"] else "off")

        if htest["heterogeneity"]:
            st.warning("Significant heterogeneity detected — the effect varies by subgroup.")
        else:
            st.success("No significant heterogeneity — the null result is uniform across subgroups.")

# ─── TAB 5: CUPED ─────────────────────────────────────────────────────────────

with tab_cuped:
    st.subheader("CUPED — Variance Reduction via Covariate Adjustment")
    st.markdown(
        "**CUPED** (Controlled-experiment Using Pre-Experiment Data) reduces estimator variance "
        "by regressing out a known covariate *X* that is correlated with the outcome *Y* "
        "but independent of the treatment assignment.\n\n"
        "$$\\hat{Y}_{\\text{cuped},i} = Y_i - \\theta\\,(X_i - \\bar{X}), "
        "\\quad \\theta = \\frac{\\operatorname{Cov}(Y_{\\text{ctrl}},\\, X_{\\text{ctrl}})}"
        "{\\operatorname{Var}(X_{\\text{ctrl}})}$$\n\n"
        "> **Note:** This dataset has no true pre-experiment covariate. We use *hour of day* "
        "as a proxy (corr ≈ 0.003 in control) to demonstrate the technique. "
        "In production, you'd use a pre-period metric such as prior-week CVR."
    )

    st.divider()

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.subheader("Variance reduction")

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("θ (adjustment coeff.)", f"{cuped['theta']:.6f}")
        vc2.metric("X̄ (mean covariate)",    f"{cuped['x_bar']:.2f} hr")
        vc3.metric("Var reduction",           f"{cuped['var_reduction_pct']:+.3f}%",
                   help="(1 − Var(Y_cuped) / Var(Y)) × 100 in the control group")

        st.divider()

        se1, se2 = st.columns(2)
        se1.metric("SE (original)",    f"{cuped['se_orig']:.6f}")
        se2.metric("SE (CUPED)",       f"{cuped['se_adj']:.6f}",
                   delta=f"{cuped['se_reduction_pct']:+.3f}%",
                   delta_color="normal")

        st.caption(
            "SE = pooled standard error of the lift estimate. "
            "With a near-zero θ the SE change is negligible — "
            "as expected when the covariate adds no information."
        )

        st.divider()

        categories = ["Original", "CUPED adjusted"]
        se_vals    = [cuped["se_orig"], cuped["se_adj"]]
        var_vals   = [cuped["var_orig"], cuped["var_adj"]]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Variance (ctrl)",
            x=categories,
            y=var_vals,
            marker_color=["#94a3b8", "#3b82f6"],
            opacity=0.85,
            text=[f"{v:.6f}" for v in var_vals],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=260,
            yaxis=dict(title="Variance of outcome"),
            margin=dict(t=20, b=20),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("Adjusted lift comparison")

        tc1, tc2 = st.columns(2)
        tc1.metric("Lift (original)",       f"{cuped['lift_orig']:+.4%}")
        tc2.metric("Lift (CUPED)",           f"{cuped['lift_adj']:+.4%}")
        tc1.metric("t-statistic",            f"{cuped['t_stat']:.4f}")
        tc2.metric("p-value (CUPED t-test)", f"{cuped['p_value']:.4f}",
                   delta="significant" if cuped["significant"] else "not significant",
                   delta_color="normal" if cuped["significant"] else "inverse")

        st.divider()

        compare_rows = {
            "Metric":            ["Control mean", "Treatment mean", "Absolute lift", "SE of lift", "p-value"],
            "Original":          [
                f"{cuped['mean_ctrl_orig']:.4%}",
                f"{cuped['mean_treat_orig']:.4%}",
                f"{cuped['lift_orig']:+.4%}",
                f"{cuped['se_orig']:.6f}",
                f"{ztest['p_value']:.4f}",
            ],
            "CUPED adjusted":    [
                f"{cuped['mean_ctrl_adj']:.4%}",
                f"{cuped['mean_treat_adj']:.4%}",
                f"{cuped['lift_adj']:+.4%}",
                f"{cuped['se_adj']:.6f}",
                f"{cuped['p_value']:.4f}",
            ],
        }
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        if abs(cuped["var_reduction_pct"]) < 0.5:
            st.info(
                "Variance reduction is near zero — the covariate (hour of day) is essentially "
                "uncorrelated with conversion in this dataset. "
                "CUPED shows its power when pre-experiment correlation is ≥ 30–40%."
            )
        elif cuped["var_reduction_pct"] > 0:
            st.success(
                f"CUPED reduced control variance by {cuped['var_reduction_pct']:.1f}%, "
                f"narrowing confidence intervals and improving sensitivity."
            )
        else:
            st.warning("Variance slightly increased — covariate is negatively correlated with outcome.")


# ─── TAB 6: Data Quality ──────────────────────────────────────────────────────

with tab_quality:
    st.subheader("Raw data audit")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw rows",        f"{info['n_total']:,}")
    q2.metric("Duplicate users", f"{info['n_duplicates']:,}",
              delta=f"{info['n_duplicates']/info['n_total']:.2%} of raw",
              delta_color="inverse")
    q3.metric("Page/group mismatches", f"{info['n_mismatches']:,}",
              delta=f"{info['n_mismatch_rate']:.2%} of raw" if 'n_mismatch_rate' in info
                    else f"{info['mismatch_rate']:.2%} of raw",
              delta_color="inverse")
    q4.metric("Clean rows", f"{len(clean):,}",
              delta=f"−{info['n_total']-len(clean):,} removed")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Group balance (clean data)")
        ctrl_n  = naive["n_control"]
        treat_n = naive["n_treatment"]
        total   = ctrl_n + treat_n
        fig = go.Figure(go.Pie(
            labels=["Control", "Treatment"],
            values=[ctrl_n, treat_n],
            hole=0.55,
            marker=dict(colors=["#3b82f6", "#f97316"]),
            textinfo="label+percent",
            hovertemplate="%{label}<br>n = %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(height=240, margin=dict(t=10, b=10),
                          showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Conversions by hour (clean data)")
        from analysis import add_time_features
        df_h = add_time_features(clean.copy())
        hourly = (
            df_h.groupby(["hour", "group"])["converted"]
            .mean().unstack("group").reset_index()
        )
        fig = go.Figure()
        for grp, color in [("control", "#3b82f6"), ("treatment", "#f97316")]:
            if grp in hourly.columns:
                fig.add_trace(go.Scatter(
                    x=hourly["hour"], y=hourly[grp],
                    name=grp.capitalize(), mode="lines+markers",
                    line=dict(color=color, width=2),
                    hovertemplate="Hour %{x}<br>CVR: %{y:.2%}<extra></extra>",
                ))
        fig.update_layout(
            height=240,
            xaxis=dict(title="Hour of day (UTC)"),
            yaxis=dict(tickformat=".1%", title="Conversion rate"),
            legend=dict(orientation="h", y=1.05),
            margin=dict(t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    with st.expander("🔍 Raw data sample (first 200 rows)"):
        st.dataframe(raw.head(200), use_container_width=True)
