"""
Rossmann Sales Dashboard - Callback Functions.

Contains all callback functions for the Dash dashboard interactivity:
- Home page: Sales trend charts and KPI updates
- Stores page: Store comparison and details
- Predict page: Sales prediction API calls
- Metrics page: MLflow metrics refresh
"""

import os
import json
from datetime import datetime
from dash import Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import mlflow

# Import app and data cache from main app module
from src.dash_app.app import app, DATA_CACHE, API_URL, MLFLOW_URI
from src.dash_app.layouts import PAGE_LAYOUTS


# ============================================================
# HOME PAGE CALLBACKS
# ============================================================

@callback(
    [
        Output("kpi-total", "children"),
        Output("kpi-avg", "children"),
        Output("kpi-stores", "children"),
        Output("kpi-promo", "children"),
        Output("sales-trend-chart", "figure"),
        Output("promo-effect-chart", "figure"),
    ],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("store-type-filter", "value"),
        Input("promo-filter", "value"),
        Input("agg-filter", "value"),
    ],
)
def update_home_charts(start_date, end_date, store_type, promo, agg_freq):
    """Update home page KPIs and charts."""
    df = DATA_CACHE["sales"].copy()

    # Filter by date
    if start_date and end_date:
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Filter by store type
    if store_type != "all":
        store_ids = DATA_CACHE["stores"][DATA_CACHE["stores"]["StoreType"] == store_type]["Store"].tolist()
        df = df[df["Store"].isin(store_ids)]

    # Filter by promo
    if promo != "all":
        df = df[df["Promo"] == promo]

    # KPIs
    total_sales = f"€{df['Sales'].sum():,.0f}"
    avg_sales = f"€{df['Sales'].mean():,.0f}"
    total_stores = f"{df['Store'].nunique()}"
    promo_rate = f"{(df['Promo'].mean() * 100):.1f}%"

    # Aggregate
    if agg_freq != "D":
        df = df.set_index("Date").groupby("Store").resample(agg_freq)["Sales"].sum().reset_index()

    # Sales trend chart
    trend_fig = px.line(
        df.groupby("Date")["Sales"].sum().reset_index(),
        x="Date",
        y="Sales",
        title="Aggregated Daily Sales",
    )
    trend_fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

    # Promo effect chart
    promo_fig = px.box(
        df,
        x="Promo",
        y="Sales",
        title="Sales Distribution: With vs Without Promotion",
        labels={"Promo": "Promotion", "Sales": "Sales (€)"},
    )
    promo_fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

    return total_sales, avg_sales, total_stores, promo_rate, trend_fig, promo_fig


# ============================================================
# STORES PAGE CALLBACKS
# ============================================================

@callback(
    [Output("store-comparison-chart", "figure"), Output("store-details-table", "children")],
    [Input("store-selector", "value"), Input("store-metric", "value")],
)
def update_store_comparison(selected_stores, metric):
    """Update store comparison chart and details table."""
    if not selected_stores:
        return go.Figure(), "Select stores to compare"

    df = DATA_CACHE["sales"][DATA_CACHE["sales"]["Store"].isin(selected_stores)].copy()

    if metric == "Sales":
        agg_df = df.groupby("Store")["Sales"].sum().reset_index()
        title = "Total Sales by Store"
    elif metric == "AvgSales":
        agg_df = df.groupby("Store")["Sales"].mean().reset_index()
        title = "Average Daily Sales by Store"
    else:
        agg_df = df.groupby("Store")["Sales"].std().reset_index()
        title = "Sales Standard Deviation by Store"

    # Bar chart
    fig = px.bar(
        agg_df,
        x="Store",
        y=metric,
        title=title,
        color="Store",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)

    # Details table
    store_details = DATA_CACHE["stores"][DATA_CACHE["stores"]["Store"].isin(selected_stores)]
    table = dbc.Table.from_dataframe(store_details, striped=True, bordered=True, hover=True, responsive=True)

    return fig, table


# ============================================================
# PREDICT PAGE CALLBACKS
# ============================================================

@callback(
    [Output("predicted-sales", "children"), Output("prediction-input-summary", "children")],
    Input("predict-btn", "n_clicks"),
    [
        State("pred-store-id", "value"),
        State("pred-day-of-week", "value"),
        State("pred-promo", "value"),
        State("pred-school-holiday", "value"),
        State("pred-store-type", "value"),
        State("pred-assortment", "value"),
        State("pred-competition-distance", "value"),
        State("pred-promo2", "value"),
    ],
    prevent_initial_call=True,
)
def predict_sales(n_clicks, store_id, day_of_week, promo, school_holiday, store_type, assortment, competition_distance, promo2):
    """Call FastAPI prediction endpoint."""
    if not n_clicks:
        raise PreventUpdate

    payload = {
        "Store": store_id,
        "DayOfWeek": day_of_week,
        "Promo": promo,
        "SchoolHoliday": school_holiday,
        "StoreType": store_type,
        "Assortment": assortment,
        "CompetitionDistance": float(competition_distance),
        "Promo2": promo2,
    }

    # Input summary
    summary = "\n".join([f"{k}: {v}" for k, v in payload.items()])

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            predicted = result.get("predicted_sales", 0)
            return f"€{predicted:,.2f}", summary
        else:
            return f"Error: {response.text}", summary
    except Exception as e:
        return f"Connection Error: {e}", summary


# ============================================================
# METRICS PAGE CALLBACKS
# ============================================================

@callback(
    [
        Output("mlflow-status", "children"),
        Output("metric-mae", "children"),
        Output("metric-rmse", "children"),
        Output("metric-r2", "children"),
        Output("metric-best-run", "children"),
        Output("feature-importance-chart", "figure"),
        Output("mlflow-runs-table", "children"),
    ],
    Input("refresh-mlflow", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_mlflow_metrics(n_clicks):
    """Fetch metrics from MLflow tracking server."""
    if not n_clicks:
        raise PreventUpdate

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("rossmann_sales")
        if not experiment:
            return "Experiment not found", "", "", "", "", go.Figure(), "No experiment found"

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mae ASC"],
            max_results=10,
        )

        if not runs:
            return "No runs found", "", "", "", "", go.Figure(), "No runs found"

        # Best run metrics
        best = runs[0]
        mae = f"{best.data.metrics.get('mae', 0):.2f}"
        rmse = f"{best.data.metrics.get('rmse', 0):.2f}"
        r2 = f"{best.data.metrics.get('r2', 0):.3f}"
        best_run = f"Run: {best.info.run_id[:8]}"

        # Feature importance (mock for now)
        features = ["Store", "Promo", "DayOfWeek", "CompetitionDistance", "StoreType", "Assortment"]
        importance = np.random.dirichlet(np.ones(len(features)), size=1)[0]
        importance = np.sort(importance)[::-1]

        fi_fig = px.bar(
            x=importance[:15],
            y=features[:15],
            orientation="h",
            title="Feature Importance (mock data)",
        )
        fi_fig.update_layout(template="plotly_dark", margin=dict(l=100, r=20, t=40, b=20))

        # Runs table
        run_data = []
        for run in runs[:5]:
            run_data.append(
                {
                    "Run ID": run.info.run_id[:8],
                    "MAE": f"{run.data.metrics.get('mae', 0):.2f}",
                    "RMSE": f"{run.data.metrics.get('rmse', 0):.2f}",
                    "n_estimators": run.data.params.get("n_estimators", "N/A"),
                    "max_depth": run.data.params.get("max_depth", "N/A"),
                }
            )
        runs_df = pd.DataFrame(run_data)
        table = dbc.Table.from_dataframe(runs_df, striped=True, bordered=True, hover=True, responsive=True)

        return "Connected", mae, rmse, r2, best_run, fi_fig, table

    except Exception as e:
        return f"Error: {e}", "", "", "", "", go.Figure(), f"Error: {e}"