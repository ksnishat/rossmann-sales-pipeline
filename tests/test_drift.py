"""Tests for PSI drift detection.

The important properties are: identical distributions give PSI 0, a real shift
gives a large PSI, degenerate inputs do not raise, and the monitor only reports
once it has enough samples to be meaningful.
"""

import json

import numpy as np
import pytest

from src.app.drift import (
    DriftMonitor,
    classify_psi,
    compute_psi,
)


class TestComputePsi:
    def test_identical_distributions_give_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(0, 1, 5000)
        assert compute_psi(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_small_shift_is_stable(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, 5000)
        shifted = rng.normal(0.1, 1, 5000)
        psi = compute_psi(base, shifted)
        assert psi < 0.10, f"a 0.1-sigma shift should be stable, got {psi}"

    def test_large_shift_is_significant(self):
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, 5000)
        shifted = rng.normal(1.0, 1, 5000)
        psi = compute_psi(base, shifted)
        assert psi >= 0.25, f"a 1-sigma shift should be significant, got {psi}"

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, 2000)
        for _ in range(5):
            assert compute_psi(base, rng.normal(0, 1, 2000)) >= 0.0

    def test_constant_feature_does_not_raise(self):
        """A sensor that never varies collapses the quantile bins."""
        assert compute_psi(np.ones(100), np.ones(100)) == 0.0

    def test_empty_input_returns_zero(self):
        assert compute_psi([], np.arange(100.0)) == 0.0
        assert compute_psi(np.arange(100.0), []) == 0.0

    def test_nan_values_are_ignored(self):
        base = np.concatenate([np.arange(100.0), [np.nan, np.inf]])
        assert compute_psi(base, np.arange(100.0)) == pytest.approx(0.0, abs=1e-9)


class TestClassifyPsi:
    @pytest.mark.parametrize(
        "psi,expected",
        [
            (0.0, "stable"),
            (0.09, "stable"),
            (0.10, "moderate"),
            (0.24, "moderate"),
            (0.25, "significant"),
            (5.0, "significant"),
        ],
    )
    def test_thresholds(self, psi, expected):
        assert classify_psi(psi) == expected

    def test_non_finite_is_unknown(self):
        assert classify_psi(float("nan")) == "unknown"


class TestDriftMonitor:
    def _baseline(self):
        rng = np.random.default_rng(7)
        return {"s1": rng.normal(0, 1, 1000).tolist()}

    def test_disabled_without_baseline(self):
        monitor = DriftMonitor(baseline={})
        assert monitor.enabled is False
        assert monitor.compute() == {}

    def test_enabled_with_baseline(self):
        assert DriftMonitor(baseline=self._baseline()).enabled is True

    def test_requires_minimum_samples(self):
        """PSI on a handful of points is noise, so it must not be reported."""
        monitor = DriftMonitor(baseline=self._baseline())
        for _ in range(5):
            monitor.observe({"s1": 0.0})
        assert monitor.compute(min_samples=30) == {}

    def test_reports_after_enough_samples(self):
        monitor = DriftMonitor(baseline=self._baseline())
        rng = np.random.default_rng(8)
        for value in rng.normal(0, 1, 200):
            monitor.observe({"s1": float(value)})
        results = monitor.compute(min_samples=30)
        assert "s1" in results
        assert results["s1"]["status"] == "stable"

    def test_detects_a_real_shift(self):
        monitor = DriftMonitor(baseline=self._baseline())
        rng = np.random.default_rng(9)
        for value in rng.normal(3.0, 1, 300):
            monitor.observe({"s1": float(value)})
        results = monitor.compute(min_samples=30)
        assert results["s1"]["status"] == "significant"
        assert monitor.max_psi(min_samples=30) >= 0.25

    def test_unknown_features_are_ignored(self):
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.observe({"not_a_sensor": 1.0})
        assert monitor.compute(min_samples=1) == {}

    def test_non_numeric_values_are_ignored(self):
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.observe({"s1": "not a number"})
        assert monitor.compute(min_samples=1) == {}

    def test_window_is_bounded(self):
        monitor = DriftMonitor(baseline=self._baseline(), window_size=10)
        for i in range(100):
            monitor.observe({"s1": float(i)})
        assert len(monitor._observed["s1"]) == 10

    def test_reset_keeps_baseline(self):
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.observe({"s1": 1.0})
        monitor.reset()
        assert monitor.enabled is True
        assert monitor.compute(min_samples=1) == {}

    def test_round_trip_through_file(self, tmp_path):
        path = tmp_path / "baseline.json"
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.save_baseline(str(path))

        reloaded = DriftMonitor.from_file(str(path))
        assert reloaded.enabled is True
        assert set(reloaded.baseline) == {"s1"}

    def test_missing_file_yields_disabled_monitor(self, tmp_path):
        monitor = DriftMonitor.from_file(str(tmp_path / "absent.json"))
        assert monitor.enabled is False

    def test_saved_baseline_is_valid_json(self, tmp_path):
        path = tmp_path / "baseline.json"
        DriftMonitor(baseline=self._baseline()).save_baseline(str(path))
        payload = json.loads(path.read_text())
        assert "features" in payload
        assert "s1" in payload["features"]


class TestApiDriftWiring:
    """The Rossmann API observes request features, not a DataFrame."""

    def _monitor(self):
        from src.app.drift import DriftMonitor

        rng = np.random.default_rng(21)
        return DriftMonitor(
            baseline={
                "Store": rng.integers(1, 1000, 1000).astype(float).tolist(),
                "Promo": rng.integers(0, 2, 1000).astype(float).tolist(),
            }
        )

    def test_numeric_features_are_observed(self, monkeypatch):
        from src.app import api

        monitor = self._monitor()
        monkeypatch.setattr(api, "_drift_monitor", monitor)

        result = api.observe_drift({"Store": 500, "Promo": 1, "StoreType": "c"})
        assert result["recorded"] is True
        # StoreType is a string and must be skipped, not crash the observer.
        assert len(monitor._observed["Store"]) == 1
        assert len(monitor._observed["Promo"]) == 1

    def test_disabled_monitor_returns_early(self, monkeypatch):
        from src.app import api

        monkeypatch.setattr(api, "_drift_monitor", None)
        assert api.observe_drift({"Store": 1}) == {"enabled": False}

    def test_drift_endpoint_reports_insufficient_data(self, monkeypatch):
        from src.app import api

        monkeypatch.setattr(api, "_drift_monitor", self._monitor())
        body = api.drift_report()
        assert body["enabled"] is True
        assert body["status"] == "insufficient_data"

    def test_drift_endpoint_disabled_without_baseline(self, monkeypatch):
        from src.app import api

        monkeypatch.setattr(api, "_drift_monitor", None)
        assert api.drift_report()["enabled"] is False
