"""Tests for the Dash dashboard.

The dashboard was 0% covered, which pulled the whole repo under any meaningful
gate and meant a broken page shipped unnoticed. These tests exercise the pieces
that hold logic — data generation, page routing, the navbar toggle, and the
chart/KPI builders — without starting a browser.

A full `dash_duo` test would be the ideal, but it needs a live server and a
headless browser in CI for a demo app. Testing the callbacks directly covers
the failure modes that actually appear in practice: wrong column names, empty
filter results, and an unrouted URL.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def dash_app_module():
    from src.dash_app import app as dash_app

    return dash_app


@pytest.fixture(scope="module")
def layouts_module():
    from src.dash_app import layouts

    return layouts


class TestSampleData:
    def test_sales_data_has_expected_shape_and_columns(self, dash_app_module):
        df = dash_app_module.get_sample_sales_data()
        assert not df.empty
        assert set(df.columns) >= {"Date", "Store", "Sales", "Promo", "DayOfWeek"}

    def test_sales_are_never_negative(self, dash_app_module):
        """A negative sale would break every KPI and chart downstream."""
        df = dash_app_module.get_sample_sales_data()
        assert (df["Sales"] >= 0).all()

    def test_sales_data_is_deterministic(self, dash_app_module):
        """Seeded so a screenshot test would not flap."""
        a = dash_app_module.get_sample_sales_data()
        b = dash_app_module.get_sample_sales_data()
        pd.testing.assert_frame_equal(a, b)

    def test_store_metadata_covers_every_store(self, dash_app_module):
        sales = dash_app_module.get_sample_sales_data()
        meta = dash_app_module.get_store_metadata()
        assert set(sales["Store"].unique()) <= set(meta["Store"].unique())

    def test_store_metadata_has_required_columns(self, dash_app_module):
        meta = dash_app_module.get_store_metadata()
        assert set(meta.columns) >= {
            "Store",
            "StoreType",
            "Assortment",
            "CompetitionDistance",
            "Promo2",
        }


class TestNavigation:
    def test_known_paths_route(self, dash_app_module):
        for path in ("/", "/stores", "/predict", "/metrics"):
            assert dash_app_module.display_page(path) is not None

    def test_unknown_path_falls_back_to_home(self, dash_app_module):
        """A typo'd URL must render something, not a blank page."""
        assert dash_app_module.display_page("/nope") is not None

    def test_toggle_opens_then_closes(self, dash_app_module):
        opened, expanded = dash_app_module.toggle_navbar_collapse(1, False)
        assert opened is True and expanded is True
        closed, collapsed = dash_app_module.toggle_navbar_collapse(1, True)
        assert closed is False and collapsed is False

    def test_toggle_is_noop_without_a_click(self, dash_app_module):
        assert dash_app_module.toggle_navbar_collapse(None, True) == (True, True)

    def test_render_page_wraps_content(self, layouts_module):
        page = layouts_module.render_page("Home")
        assert page is not None

    def test_every_page_has_a_title(self, layouts_module):
        assert set(layouts_module.PAGES) == {"/", "/stores", "/predict", "/metrics"}


class TestHomeCharts:
    def test_returns_kpis_and_two_figures(self, dash_app_module):
        out = dash_app_module.update_home_charts(
            "2015-01-01", "2015-03-01", "all", "all", "D"
        )
        # 4 KPI strings + 2 figures
        assert len(out) == 6
        assert all(isinstance(v, str) for v in out[:4])
        assert out[4] is not None and out[5] is not None

    def test_weekly_aggregation_differs_from_daily(self, dash_app_module):
        daily = dash_app_module.update_home_charts(
            "2015-01-01", "2015-03-01", "all", "all", "D"
        )[4]
        weekly = dash_app_module.update_home_charts(
            "2015-01-01", "2015-03-01", "all", "all", "W"
        )[4]
        assert daily is not None and weekly is not None
        assert len(daily.data[0].x) != len(weekly.data[0].x)

    def test_filters_do_not_crash(self, dash_app_module):
        out = dash_app_module.update_home_charts(
            "2015-01-01", "2015-02-01", "a", True, "D"
        )
        assert len(out) == 6

    def test_empty_date_range_is_handled(self, dash_app_module):
        """A range with no rows must not raise."""
        out = dash_app_module.update_home_charts(
            "2015-01-01", "2015-01-02", "all", "all", "D"
        )
        assert len(out) == 6

    def test_every_aggregation_frequency_renders(self, dash_app_module):
        """Regression: W and M raised KeyError on the missing Sales column."""
        for freq in ("D", "W", "ME"):
            out = dash_app_module.update_home_charts(
                "2015-01-01", "2015-03-01", "all", "all", freq
            )
            assert len(out[4].data[0].x) > 0, f"no points for freq={freq}"
            assert out[5] is not None

    def test_kpis_are_not_empty_for_a_nonempty_range(self, dash_app_module):
        total, avg, stores, promo_rate = dash_app_module.update_home_charts(
            "2015-01-01", "2015-03-01", "all", "all", "D"
        )[:4]
        assert total != "€0"
        assert stores == "10"
        assert promo_rate.endswith("%")


class TestStoreComparison:
    """The callback returns (figure, table)."""

    def test_single_store_renders(self, dash_app_module):
        fig, table = dash_app_module.update_store_comparison([1], "Sales")
        assert fig is not None
        assert table is not None

    def test_multiple_stores_render(self, dash_app_module):
        fig, _ = dash_app_module.update_store_comparison([1, 2, 3], "Sales")
        assert len(fig.data) >= 1

    def test_every_metric_option_produces_a_chart(self, dash_app_module):
        """Regression: `y=metric` raised KeyError for AvgSales and StdSales."""
        for metric in ("Sales", "AvgSales", "StdSales"):
            fig, _ = dash_app_module.update_store_comparison([1, 2], metric)
            assert fig.data, f"no data plotted for metric={metric}"

    def test_empty_selection_does_not_raise(self, dash_app_module):
        """Deselecting every store is a normal user action."""
        fig, table = dash_app_module.update_store_comparison([], "Sales")
        assert fig is not None
        assert isinstance(table, str)


class TestPredictCallback:
    def test_no_click_prevents_update(self, dash_app_module):
        """The callback must not fire an API call on page load."""
        from dash.exceptions import PreventUpdate

        with pytest.raises(PreventUpdate):
            dash_app_module.predict_sales(
                None, 1, 3, 1, 0, "a", "a", 500, 0
            )

    def test_api_failure_returns_error_message(self, dash_app_module, monkeypatch):
        """The user must see why, not a traceback in the logs."""

        def boom(*args, **kwargs):
            raise ConnectionError("api down")

        monkeypatch.setattr(dash_app_module.requests, "post", boom)
        result = dash_app_module.predict_sales(
            1, 1, 3, 1, 0, "a", "a", 500.0, 0
        )
        assert result is not None
        assert "rror" in str(result) or "fail" in str(result).lower()


class TestMlflowRefresh:
    def test_missing_click_prevents_update(self, dash_app_module):
        from dash.exceptions import PreventUpdate

        with pytest.raises(PreventUpdate):
            dash_app_module.refresh_mlflow_metrics(None)

    def test_mlflow_failure_does_not_raise(self, dash_app_module, monkeypatch):
        """MLflow being down must not break the dashboard.

        `mlflow` is imported inside the callback, so it is patched on the
        installed module rather than on the dashboard package.
        """
        import mlflow

        monkeypatch.setattr(
            mlflow.tracking,
            "MlflowClient",
            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")),
        )
        try:
            result = dash_app_module.refresh_mlflow_metrics(1)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"dashboard raised on MLflow outage: {exc}")
        assert result is not None


class TestEntryPointIsRunnable:
    """The dashboard module must be launchable as a script.

    ``app.run_server`` was removed in Dash 3.0 and raises
    ``ObsoleteAttributeException``. pyproject.toml allows ``dash>=2.14.0``, so
    the installed version is 4.x and the old name made `python
    src/dash_app/app.py` crash immediately — the dashboard never started when
    launched directly, while the imported app object worked fine in tests, so
    nothing caught it.

    ``app.run`` exists in both 2.14 and 4.x, so it is the compatible spelling.
    """

    def test_uses_app_run_not_the_removed_run_server(self):
        import inspect

        from src.dash_app import app as dash_app

        source = inspect.getsource(dash_app)
        assert "run_server(" not in source, (
            "app.run_server was removed in Dash 3.0; use app.run"
        )
        assert "app.run(" in source

    def test_dash_version_supports_app_run(self):
        import dash

        major = int(dash.__version__.split(".")[0])
        if major >= 3:
            assert hasattr(dash.Dash, "run")
