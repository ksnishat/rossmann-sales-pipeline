"""
Rossmann Sales Analytics Dashboard - Plotly Dash Application.

Multi-page dashboard for retail sales forecasting:
- Home: Sales trends and overview
- Stores: Store comparison and performance
- Predict: Sales prediction form
- Metrics: Model performance and monitoring

Run: python -m src.dash_app.app
"""

import os
import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests

# Import layout components
from src.dash_app.layouts import app, render_page, NAVBAR, PAGES

# API base URL
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Global data cache
DATA_CACHE = {}

# Sample data for demo
def get_sample_sales_data():
    """Generate sample sales data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2015-07-31", freq="D")
    stores = list(range(1, 11))

    data = []
    for store in stores:
        for date in dates:
            base_sales = np.random.normal(7000, 1500)
            promo_boost = np.random.choice([0, 800], p=[0.7, 0.3])
            weekend_boost = 1000 if date.weekday() >= 5 else 0
            sales = max(0, base_sales + promo_boost + weekend_boost)

            data.append(
                {
                    "Date": date,
                    "Store": store,
                    "Sales": sales,
                    "Promo": promo_boost > 0,
                    "DayOfWeek": date.weekday() + 1,
                }
            )

    return pd.DataFrame(data)


def get_store_metadata():
    """Generate sample store metadata."""
    return pd.DataFrame(
        {
            "Store": list(range(1, 11)),
            "StoreType": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "d"],
            "Assortment": ["a", "a", "b", "b", "c", "c", "a", "b", "c", "a"],
            "CompetitionDistance": [1200, 800, 2000, 1500, 500, 1800, 900, 2500, 700, 1100],
            "Promo2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


# Load initial data
DATA_CACHE["sales"] = get_sample_sales_data()
DATA_CACHE["stores"] = get_store_metadata()


# ============================================================
# PAGE CONTENT
# ============================================================

# Home Page Layout
HOME_LAYOUT = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Sales Overview", className="mb-3"),
                        html.P(
                            "Daily sales trends across all Rossmann stores. "
                            "Use the controls to filter by date range, store type, and promotion status.",
                            className="text-muted mb-4",
                        ),
                    ],
                    width=12,
                ),
            ],
            className="mb-3",
        ),
        # Filters
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Date Range"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=DATA_CACHE["sales"]["Date"].min(),
                            end_date=DATA_CACHE["sales"]["Date"].max(),
                            display_format="YYYY-MM-DD",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("Store Type"),
                        dcc.Dropdown(
                            id="store-type-filter",
                            options=[
                                {"label": "All Types", "value": "all"},
                                {"label": "Type a", "value": "a"},
                                {"label": "Type b", "value": "b"},
                                {"label": "Type c", "value": "c"},
                                {"label": "Type d", "value": "d"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("Promotion"),
                        dcc.Dropdown(
                            id="promo-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "With Promo", "value": 1},
                                {"label": "Without Promo", "value": 0},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("Aggregation"),
                        dcc.Dropdown(
                            id="agg-filter",
                            options=[
                                {"label": "Daily", "value": "D"},
                                {"label": "Weekly", "value": "W"},
                                {"label": "Monthly", "value": "M"},
                            ],
                            value="D",
                            clearable=False,
                        ),
                    ],
                    width=3,
                ),
            ],
            className="mb-3",
        ),
        # KPI Cards
        dbc.Row(
            [
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Sales", className="card-title"), html.H2(id="kpi-total", className="text-primary")])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Avg Daily Sales", className="card-title"), html.H2(id="kpi-avg", className="text-success")])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Stores", className="card-title"), html.H2(id="kpi-stores", className="text-info")])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Promo Rate", className="card-title"), html.H2(id="kpi-promo", className="text-warning")])), width=3),
            ],
            className="mb-4",
        ),
        # Sales Trend Chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Daily Sales Trend", className="card-title"),
                                dcc.Loading(dcc.Graph(id="sales-trend-chart")),
                            ]
                        )
                    ),
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        # Promo Effect Chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Promotion Effect on Sales", className="card-title"),
                                dcc.Loading(dcc.Graph(id="promo-effect-chart")),
                            ]
                        )
                    ),
                    width=12,
                ),
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)


# Stores Page Layout
STORES_LAYOUT = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Store Comparison", className="mb-3"),
                        html.P(
                            "Compare performance across individual stores. "
                            "Click on a store in the chart to see details.",
                            className="text-muted mb-4",
                        ),
                    ],
                    width=12,
                ),
            ],
            className="mb-3",
        ),
        # Store selector
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select Stores"),
                        dcc.Dropdown(
                            id="store-selector",
                            options=[
                                {"label": f"Store {s}", "value": s}
                                for s in DATA_CACHE["stores"]["Store"].unique()
                            ],
                            value=[1, 2, 3],
                            multi=True,
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Label("Metric"),
                        dcc.Dropdown(
                            id="store-metric",
                            options=[
                                {"label": "Total Sales", "value": "Sales"},
                                {"label": "Average Sales", "value": "AvgSales"},
                                {"label": "Sales Std Dev", "value": "StdSales"},
                            ],
                            value="Sales",
                            clearable=False,
                        ),
                    ],
                    width=3,
                ),
            ],
            className="mb-3",
        ),
        # Store comparison chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Store Performance Comparison", className="card-title"),
                                dcc.Loading(dcc.Graph(id="store-comparison-chart")),
                            ]
                        )
                    ),
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        # Store details table
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Store Details", className="card-title"),
                                html.Div(id="store-details-table"),
                            ]
                        )
                    ),
                    width=12,
                ),
            ],
        ),
    ],
    fluid=True,
)


# Predict Page Layout
PREDICT_LAYOUT = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Sales Prediction", className="mb-3"),
                        html.P(
                            "Enter store details to predict daily sales using the trained Random Forest model. "
                            "The model uses historical data, promotions, holidays, and competitor information.",
                            className="text-muted mb-4",
                        ),
                    ],
                    width=12,
                ),
            ],
            className="mb-3",
        ),
        # Prediction Form
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Store Information", className="card-title mb-3"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Store ID"),
                                                dbc.Input(
                                                    id="pred-store-id", type="number", value=1115, min=1, max=1115
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Day of Week"),
                                                dbc.Select(
                                                    id="pred-day-of-week",
                                                    options=[
                                                        {"label": "Monday", "value": 1},
                                                        {"label": "Tuesday", "value": 2},
                                                        {"label": "Wednesday", "value": 3},
                                                        {"label": "Thursday", "value": 4},
                                                        {"label": "Friday", "value": 5},
                                                        {"label": "Saturday", "value": 6},
                                                        {"label": "Sunday", "value": 7},
                                                    ],
                                                    value=5,
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Promotion"),
                                                dbc.RadioItems(
                                                    id="pred-promo",
                                                    options=[
                                                        {"label": "Yes", "value": 1},
                                                        {"label": "No", "value": 0},
                                                    ],
                                                    value=0,
                                                    inline=True,
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("School Holiday"),
                                                dbc.RadioItems(
                                                    id="pred-school-holiday",
                                                    options=[
                                                        {"label": "Yes", "value": 1},
                                                        {"label": "No", "value": 0},
                                                    ],
                                                    value=0,
                                                    inline=True,
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Store Type"),
                                                dbc.Select(
                                                    id="pred-store-type",
                                                    options=[
                                                        {"label": "Type a", "value": "a"},
                                                        {"label": "Type b", "value": "b"},
                                                        {"label": "Type c", "value": "c"},
                                                        {"label": "Type d", "value": "d"},
                                                    ],
                                                    value="a",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Assortment"),
                                                dbc.Select(
                                                    id="pred-assortment",
                                                    options=[
                                                        {"label": "Level a", "value": "a"},
                                                        {"label": "Level b", "value": "b"},
                                                        {"label": "Level c", "value": "c"},
                                                    ],
                                                    value="a",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Competition Distance (m)"),
                                                dbc.Input(
                                                    id="pred-competition-distance",
                                                    type="number",
                                                    value=500,
                                                    min=0,
                                                    step=100,
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Promo2 Active"),
                                                dbc.RadioItems(
                                                    id="pred-promo2",
                                                    options=[
                                                        {"label": "Yes", "value": 1},
                                                        {"label": "No", "value": 0},
                                                    ],
                                                    value=0,
                                                    inline=True,
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Button(
                                    "Predict Sales",
                                    id="predict-btn",
                                    color="primary",
                                    size="lg",
                                    className="w-100",
                                ),
                            ]
                        )
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Prediction Result", className="card-title mb-3"),
                                html.Div(
                                    id="prediction-result",
                                    children=[
                                        html.H3("€ —", className="text-center text-muted", id="predicted-sales"),
                                        html.P("Enter store details and click Predict", className="text-center text-muted"),
                                    ],
                                ),
                                html.Hr(),
                                html.H6("Input Summary", className="mt-3"),
                                html.Pre(id="prediction-input-summary", style={"fontSize": "0.85rem"}),
                            ]
                        )
                    ),
                    width=6,
                ),
            ],
        ),
    ],
    fluid=True,
)


# Metrics Page Layout
METRICS_LAYOUT = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Model Metrics & Monitoring", className="mb-3"),
                        html.P(
                            "MLflow experiment tracking and model performance metrics. "
                            "Refresh to pull latest metrics from MLflow server.",
                            className="text-muted mb-4",
                        ),
                    ],
                    width=12,
                ),
            ],
            className="mb-3",
        ),
        # MLflow connection status
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Refresh MLflow",
                            id="refresh-mlflow",
                            color="secondary",
                            className="me-2",
                        ),
                        html.Span(id="mlflow-status", className="ms-2"),
                    ],
                    width=12,
                ),
            ],
            className="mb-3",
        ),
        # Model metrics cards
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("MAE", className="card-title"),
                                html.H2(id="metric-mae", className="text-primary"),
                            ]
                        )
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("RMSE", className="card-title"),
                                html.H2(id="metric-rmse", className="text-success"),
                            ]
                        )
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("R² Score", className="card-title"),
                                html.H2(id="metric-r2", className="text-info"),
                            ]
                        )
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Best Run", className="card-title"),
                                html.P(id="metric-best-run", className="text-warning"),
                            ]
                        )
                    ),
                    width=3,
                ),
            ],
            className="mb-4",
        ),
        # Feature importance chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Feature Importance (Top 15)", className="card-title"),
                                dcc.Loading(dcc.Graph(id="feature-importance-chart")),
                            ]
                        )
                    ),
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        # Experiment runs table
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Recent MLflow Runs", className="card-title"),
                                html.Div(id="mlflow-runs-table"),
                            ]
                        )
                    ),
                    width=12,
                ),
            ],
        ),
    ],
    fluid=True,
)


# ============================================================
# ROUTING
# ============================================================

# Page content mapping
PAGE_LAYOUTS = {
    "/": HOME_LAYOUT,
    "/stores": STORES_LAYOUT,
    "/predict": PREDICT_LAYOUT,
    "/metrics": METRICS_LAYOUT,
}


app.layout = render_page("/")


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    """Display the correct page based on URL."""
    return PAGE_LAYOUTS.get(pathname, HOME_LAYOUT)


@callback(
    [Output("navbar-collapse", "is_open"), Output("navbar-toggler", "aria-expanded")],
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    """Toggle navbar collapse on mobile."""
    if n:
        return not is_open, not is_open
    return is_open, is_open


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
        import mlflow
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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("DASH_PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "true").lower() == "true"
    app.run_server(host="0.0.0.0", port=port, debug=debug)