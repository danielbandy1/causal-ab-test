"""Tests for src/analysis.py."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    audit_data,
    clean_data,
    naive_comparison,
    ztest_conversion,
    power_analysis,
    add_time_features,
    subgroup_analysis,
    heterogeneity_test,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """1,000-row perfectly clean A/B dataset (no mismatches, no dupes)."""
    rng = np.random.default_rng(0)
    n = 1_000
    groups = ["control"] * 500 + ["treatment"] * 500
    pages  = ["old_page"] * 500 + ["new_page"] * 500
    conv   = list(rng.binomial(1, 0.12, 500)) + list(rng.binomial(1, 0.14, 500))
    ts     = pd.date_range("2017-01-02", periods=n, freq="10min")
    return pd.DataFrame({
        "user_id":      range(n),
        "timestamp":    ts,
        "group":        groups,
        "landing_page": pages,
        "converted":    conv,
    })


@pytest.fixture
def dirty_df(clean_df):
    """Introduces 50 mismatches and 30 duplicate user IDs."""
    df = clean_df.copy()
    # Mismatches: flip landing_page for first 50 treatment rows
    treat_idx = df[df["group"] == "treatment"].index[:50]
    df.loc[treat_idx, "landing_page"] = "old_page"
    # Duplicates: repeat first 30 rows with new timestamps
    dupes = df.iloc[:30].copy()
    dupes["timestamp"] = dupes["timestamp"] + pd.Timedelta("1h")
    return pd.concat([df, dupes], ignore_index=True)


# ── audit_data ────────────────────────────────────────────────────────────────

def test_audit_clean_data(clean_df):
    result = audit_data(clean_df)
    assert result["n_total"] == 1_000
    assert result["n_duplicates"] == 0
    assert result["n_mismatches"] == 0
    assert 0 < result["conversion_overall"] < 1


def test_audit_detects_duplicates(dirty_df):
    result = audit_data(dirty_df)
    assert result["n_duplicates"] == 30


def test_audit_detects_mismatches(dirty_df):
    result = audit_data(dirty_df)
    assert result["n_mismatches"] == 50


def test_audit_group_balance(clean_df):
    result = audit_data(clean_df)
    assert set(result["group_balance"].keys()) == {"control", "treatment"}
    assert result["group_balance"]["control"] == 500
    assert result["group_balance"]["treatment"] == 500


# ── clean_data ────────────────────────────────────────────────────────────────

def test_clean_removes_mismatches(dirty_df):
    cleaned = clean_data(dirty_df)
    mismatches = (
        ((cleaned["group"] == "treatment") & (cleaned["landing_page"] == "old_page")) |
        ((cleaned["group"] == "control")   & (cleaned["landing_page"] == "new_page"))
    )
    assert mismatches.sum() == 0


def test_clean_removes_duplicates(dirty_df):
    cleaned = clean_data(dirty_df)
    assert cleaned["user_id"].duplicated().sum() == 0


def test_clean_smaller_than_dirty(dirty_df):
    cleaned = clean_data(dirty_df)
    assert len(cleaned) < len(dirty_df)


def test_clean_idempotent(clean_df):
    """Cleaning already-clean data should return the same rows."""
    cleaned = clean_data(clean_df)
    assert len(cleaned) == len(clean_df)


# ── naive_comparison ──────────────────────────────────────────────────────────

def test_naive_comparison_keys(clean_df):
    result = naive_comparison(clean_df)
    expected_keys = {"control_rate", "treatment_rate", "absolute_lift",
                     "relative_lift", "n_control", "n_treatment"}
    assert set(result.keys()) == expected_keys


def test_naive_comparison_rates_in_range(clean_df):
    result = naive_comparison(clean_df)
    assert 0 < result["control_rate"]   < 1
    assert 0 < result["treatment_rate"] < 1


def test_naive_comparison_lift_arithmetic(clean_df):
    result = naive_comparison(clean_df)
    expected_lift = round(result["treatment_rate"] - result["control_rate"], 4)
    assert result["absolute_lift"] == pytest.approx(expected_lift, abs=1e-4)


def test_naive_comparison_counts(clean_df):
    result = naive_comparison(clean_df)
    assert result["n_control"]   == 500
    assert result["n_treatment"] == 500


# ── ztest_conversion ──────────────────────────────────────────────────────────

def test_ztest_keys(clean_df):
    result = ztest_conversion(clean_df)
    assert {"z_stat", "p_value", "significant", "ci_95"} <= set(result.keys())


def test_ztest_p_value_valid(clean_df):
    result = ztest_conversion(clean_df)
    assert 0 <= result["p_value"] <= 1


def test_ztest_ci_ordered(clean_df):
    result = ztest_conversion(clean_df)
    lo, hi = result["ci_95"]
    assert lo < hi


def test_ztest_significant_for_large_effect():
    """Known-significant case: 10% vs 20% with n=2000 each."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "group":     ["control"]   * 2_000 + ["treatment"] * 2_000,
        "converted": list(rng.binomial(1, 0.10, 2_000)) + list(rng.binomial(1, 0.20, 2_000)),
    })
    result = ztest_conversion(df)
    assert result["significant"] is True
    assert result["p_value"] < 0.001


def test_ztest_not_significant_for_null_effect():
    """Known-null case: identical groups → p should be ~1.0."""
    n = 5_000
    converted = [1] * 600 + [0] * 4_400  # exactly 12%, same for both groups
    df = pd.DataFrame({
        "group":     ["control"] * n + ["treatment"] * n,
        "converted": converted + converted,
    })
    result = ztest_conversion(df)
    assert result["p_value"] > 0.99


# ── power_analysis ────────────────────────────────────────────────────────────

def test_power_analysis_keys(clean_df):
    result = power_analysis(clean_df)
    assert {"effect_size", "required_n_per_group", "actual_n_per_group",
            "actual_power", "adequately_powered"} <= set(result.keys())


def test_power_analysis_power_in_range(clean_df):
    result = power_analysis(clean_df)
    assert 0 <= result["actual_power"] <= 1


def test_power_adequately_powered_for_large_sample():
    """A very large sample detecting a non-trivial effect should be powered."""
    rng = np.random.default_rng(9)
    n = 50_000
    df = pd.DataFrame({
        "group":     ["control"]   * n + ["treatment"] * n,
        "converted": list(rng.binomial(1, 0.10, n)) + list(rng.binomial(1, 0.12, n)),
    })
    result = power_analysis(df)
    assert result["adequately_powered"] is True


# ── add_time_features ─────────────────────────────────────────────────────────

def test_add_time_features_columns(clean_df):
    result = add_time_features(clean_df)
    assert "hour"        in result.columns
    assert "day_of_week" in result.columns


def test_add_time_features_ranges(clean_df):
    result = add_time_features(clean_df)
    assert result["hour"].between(0, 23).all()
    assert result["day_of_week"].between(0, 6).all()


def test_add_time_features_no_mutation(clean_df):
    original_cols = set(clean_df.columns)
    add_time_features(clean_df)
    assert set(clean_df.columns) == original_cols  # original unchanged


# ── subgroup_analysis ─────────────────────────────────────────────────────────

def test_subgroup_analysis_returns_dataframe(clean_df):
    result = subgroup_analysis(clean_df)
    assert isinstance(result, pd.DataFrame)


def test_subgroup_analysis_columns(clean_df):
    result = subgroup_analysis(clean_df)
    expected = {"subgroup_type", "subgroup", "n_control", "n_treatment",
                "lift", "ci_low", "ci_high", "p_raw", "p_bonferroni", "significant"}
    assert expected <= set(result.columns)


def test_subgroup_analysis_lift_in_ci(clean_df):
    """Lift point estimate should fall within its own CI."""
    result = subgroup_analysis(clean_df)
    for _, row in result.iterrows():
        assert row["ci_low"] <= row["lift"] <= row["ci_high"], (
            f"Subgroup {row['subgroup']}: lift {row['lift']} outside CI "
            f"[{row['ci_low']}, {row['ci_high']}]"
        )


def test_subgroup_analysis_bonferroni_ge_raw(clean_df):
    result = subgroup_analysis(clean_df)
    assert (result["p_bonferroni"] >= result["p_raw"]).all()


# ── heterogeneity_test ────────────────────────────────────────────────────────

def test_heterogeneity_test_keys(clean_df):
    result = heterogeneity_test(clean_df)
    assert {"lr_stat", "df", "p_value", "heterogeneity"} <= set(result.keys())


def test_heterogeneity_test_p_value_valid(clean_df):
    result = heterogeneity_test(clean_df)
    assert 0 <= result["p_value"] <= 1


def test_heterogeneity_test_lr_stat_nonneg(clean_df):
    result = heterogeneity_test(clean_df)
    assert result["lr_stat"] >= 0
