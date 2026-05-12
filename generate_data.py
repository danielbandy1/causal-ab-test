#!/usr/bin/env python3
"""
Generate a synthetic A/B test dataset matching the statistical properties
of the Udacity A/B Test Results dataset (Kaggle).

Real dataset properties reproduced here:
  - ~294k rows, roughly equal control/treatment split
  - Control conversion rate:   12.04%
  - Treatment conversion rate: 11.88%  (no significant lift)
  - ~1.3% page/group mismatches (data quality issue)
  - ~1.3% duplicate user IDs
  - Timestamps span Jan–Mar 2017, volume varies by hour/day
"""

import pathlib
import numpy as np
import pandas as pd

SEED        = 42
N_USERS     = 294_478
P_CTRL      = 0.1204
P_TREAT     = 0.1188
MISMATCH_RT = 0.013
DUPE_RT     = 0.013
OUT         = pathlib.Path("data/raw/ab_data.csv")

rng = np.random.default_rng(SEED)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    user_ids = rng.choice(range(600_000, 1_000_000), size=N_USERS, replace=False)

    groups        = rng.choice(["control", "treatment"], size=N_USERS, p=[0.5002, 0.4998])
    landing_pages = np.where(groups == "control", "old_page", "new_page")

    # Inject mismatches (group/page don't match — a real data quality bug)
    n_mismatch   = int(N_USERS * MISMATCH_RT)
    mismatch_idx = rng.choice(N_USERS, size=n_mismatch, replace=False)
    landing_pages[mismatch_idx] = np.where(
        landing_pages[mismatch_idx] == "old_page", "new_page", "old_page"
    )

    conv_probs = np.where(groups == "control", P_CTRL, P_TREAT)
    converted  = rng.binomial(1, conv_probs)

    # Timestamps: Jan 2 – Mar 31 2017, heavier traffic during business hours
    start_ts = pd.Timestamp("2017-01-02")
    end_ts   = pd.Timestamp("2017-03-31 23:59:59")
    span_sec = int((end_ts - start_ts).total_seconds())

    # Weight by hour-of-day to simulate realistic traffic patterns
    hour_weights = np.array([
        0.5, 0.3, 0.2, 0.2, 0.3, 0.5,   # 00-05
        0.8, 1.5, 2.5, 3.0, 3.2, 3.0,   # 06-11
        2.8, 2.7, 2.5, 2.6, 2.8, 3.0,   # 12-17
        3.2, 3.0, 2.5, 2.0, 1.5, 1.0,   # 18-23
    ])
    hour_weights /= hour_weights.sum()

    hours     = rng.choice(24, size=N_USERS, p=hour_weights)
    mins_secs = rng.integers(0, 3600, size=N_USERS)
    day_offsets = rng.integers(0, span_sec // 86_400, size=N_USERS)

    timestamps = (
        start_ts
        + pd.to_timedelta(day_offsets, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(mins_secs, unit="s")
    )

    df = pd.DataFrame({
        "user_id":      user_ids,
        "timestamp":    timestamps,
        "group":        groups,
        "landing_page": landing_pages,
        "converted":    converted,
    }).sort_values("timestamp").reset_index(drop=True)

    # Inject duplicate user IDs (same user appears twice — another data quality bug)
    n_dupes   = int(N_USERS * DUPE_RT)
    dupe_rows = df.sample(n=n_dupes, random_state=SEED).copy()
    dupe_rows["timestamp"] = dupe_rows["timestamp"] + pd.to_timedelta(
        rng.integers(1, 7200, size=n_dupes), unit="s"
    )
    df = pd.concat([df, dupe_rows]).sort_values("timestamp").reset_index(drop=True)

    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    df.to_csv(OUT, index=False)

    ctrl  = df[df["group"] == "control"]["converted"].mean()
    treat = df[df["group"] == "treatment"]["converted"].mean()
    print(f"Wrote {len(df):,} rows to {OUT}")
    print(f"  control conversion:   {ctrl:.4f}")
    print(f"  treatment conversion: {treat:.4f}")
    print(f"  rows: {len(df):,}  |  unique users: {df['user_id'].nunique():,}")

if __name__ == "__main__":
    main()
