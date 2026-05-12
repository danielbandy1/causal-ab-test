#!/usr/bin/env python3
"""
Fetch the A/B test dataset.

Tries several known mirrors for the Udacity A/B Test Results CSV.
If all mirrors fail, falls back to generating a synthetic dataset
with identical statistical properties (same conversion rates, same
data quality issues — mismatches and duplicate user IDs).

Run:
    python3 download_data.py
"""

import pathlib
import urllib.request

MIRRORS = [
    "https://raw.githubusercontent.com/Suryaanshhh/Analyze-A-B-Test-Results/master/ab_data.csv",
    "https://raw.githubusercontent.com/vgaurav3011/Statistics-for-Machine-Learning/master/projects/AB%20Testing/ab_data.csv",
]

DEST = pathlib.Path(__file__).parent / "data" / "raw" / "ab_data.csv"


def try_download() -> bool:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS:
        try:
            print(f"Trying {url} ...")
            urllib.request.urlretrieve(url, DEST)
            size_mb = DEST.stat().st_size / 1_000_000
            if size_mb < 1:
                DEST.unlink()
                continue
            print(f"Downloaded — {size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"  failed: {e}")
    return False


def generate_fallback():
    print("All mirrors failed — generating synthetic dataset...")
    import generate_data  # noqa: PLC0415
    generate_data.main()


def main():
    if DEST.exists():
        print(f"Already present: {DEST}  ({DEST.stat().st_size/1e6:.1f} MB)")
        return
    if not try_download():
        generate_fallback()


if __name__ == "__main__":
    main()
