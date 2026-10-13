"""Tests for the DVC pipeline scripts.

These scripts were previously untested. The feature engineering script had a
closure bug where every rolling window used the final loop value.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN = REPO_ROOT / "src" / "scripts" / "clean_silver.py"
FEATURE = REPO_ROOT / "src" / "scripts" / "feature_engineer_gold.py"


def _run(script: Path, *args):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def _sales_frame(n=60, store=1):
    return pd.DataFrame(
        {
            "Store": [store] * n,
            "Date": pd.date_range("2024-01-01", periods=n),
            "Sales": [float(i) for i in range(n)],
            "Customers": [float(i) for i in range(n)],
            "Open": [1] * n,
            "Promo": [0] * n,
            "StateHoliday": ["0"] * n,
            "SchoolHoliday": [0] * n,
        }
    )


class TestCleanSilver:
    def test_writes_a_cleaned_file(self, tmp_path):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        _sales_frame().to_csv(bronze / "train.csv", index=False)
        silver = tmp_path / "silver"

        result = _run(CLEAN, "--input", str(bronze), "--output", str(silver))

        assert result.returncode == 0, result.stderr
        assert (silver / "train.csv").exists()

    def test_drops_duplicates(self, tmp_path):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        df = _sales_frame(n=10)
        pd.concat([df, df]).to_csv(bronze / "train.csv", index=False)
        silver = tmp_path / "silver"

        _run(CLEAN, "--input", str(bronze), "--output", str(silver))

        assert len(pd.read_csv(silver / "train.csv")) == 10

    def test_removes_implausible_sales(self, tmp_path):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        df = _sales_frame(n=10)
        df.loc[0, "Sales"] = 999_999.0
        df.to_csv(bronze / "train.csv", index=False)
        silver = tmp_path / "silver"

        _run(CLEAN, "--input", str(bronze), "--output", str(silver))

        cleaned = pd.read_csv(silver / "train.csv")
        assert cleaned["Sales"].max() <= 100_000

    def test_fills_missing_sales(self, tmp_path):
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        df = _sales_frame(n=10)
        df.loc[0, "Sales"] = None
        df.to_csv(bronze / "train.csv", index=False)
        silver = tmp_path / "silver"

        _run(CLEAN, "--input", str(bronze), "--output", str(silver))

        assert not pd.read_csv(silver / "train.csv")["Sales"].isnull().any()


class TestFeatureEngineerGold:
    def test_rolling_windows_are_distinct(self):
        """Each window must produce its own value, not the last window's."""
        from src.scripts.feature_engineer_gold import ROLLING_WINDOWS, create_features

        out = create_features(_sales_frame(n=100))

        means = [
            out[f"Sales_rolling_mean_{w}"].iloc[-1]
            for w in ROLLING_WINDOWS
            if f"Sales_rolling_mean_{w}" in out.columns
        ]
        assert means, "no rolling mean columns were produced"
        assert len(set(means)) == len(means), f"windows collapsed to one value: {means}"

    def test_longer_windows_are_smoother(self):
        """A longer window must average over more points, so on a linear ramp
        its final value must be lower than a shorter window's."""
        from src.scripts.feature_engineer_gold import create_features

        out = create_features(_sales_frame(n=100))

        short = out["Sales_rolling_mean_7"].iloc[-1]
        long = out["Sales_rolling_mean_90"].iloc[-1]
        assert long < short, f"expected the 90-window mean ({long}) < 7-window ({short})"

    def test_creates_lag_features(self):
        from src.scripts.feature_engineer_gold import LAGS, create_features

        out = create_features(_sales_frame(n=100))

        for lag in LAGS:
            assert f"Sales_lag_{lag}" in out.columns

    def test_creates_date_features(self):
        from src.scripts.feature_engineer_gold import create_features

        out = create_features(_sales_frame(n=100))

        for col in ("DayOfWeek", "Month", "Year", "IsWeekend"):
            assert col in out.columns

    def test_does_not_mutate_the_input(self):
        from src.scripts.feature_engineer_gold import create_features

        df = _sales_frame(n=50)
        before = df.columns.tolist()

        create_features(df)

        assert df.columns.tolist() == before
