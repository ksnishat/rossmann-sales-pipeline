"""
Dashboard app - main Dash layout file.

Provides the page structure for all Dash routes:
- / (Home: Sales Trends)
- /stores (Store Comparison)
- /predict (Sales Prediction Form)
- /metrics (Model Metrics & Performance)
"""

import dash
from dash import html
import dash_bootstrap_components as dbc

# Bootstrap theme
THEME = dbc.themes.SOLAR

# External CSS
EXTRA_STYLESHEETS = ["https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"]

app = dash.Dash(__name__, external_stylesheets=EXTRA_STYLESHEETS, theme=THEME)
app.title = "Rossmann Sales Analytics Dashboard"
server = app.server

# Page titles
PAGES = {
    "/": "Home",
    "/stores": "Stores",
    "/predict": "Predict",
    "/metrics": "Metrics",
}

# Navigation bar
NAVBAR = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            html.Div(
                                [
                                    html.I(className="fas fa-chart-line me-2"),
                                    html.Strong("Rossmann Sales AI", className="navbar-brand mb-0 h1"),
                                ],
                                className="d-flex align-items-center",
                            ),
                            href="/",
                            style={"textDecoration": "none"},
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.NavbarToggler(id="navbar-toggler"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Collapse(
                            dbc.NavbarNav(
                                [
                                    dbc.NavLink("Home", href="/", id="home", active_class="active"),
                                    dbc.NavLink("Stores", href="/stores", id="stores", active_class="active"),
                                    dbc.NavLink("Predict", href="/predict", id="predict", active_class="active"),
                                    dbc.NavLink("Metrics", href="/metrics", id="metrics", active_class="active"),
                                ],
                                className="ms-auto",
                                navbar=True,
                            ),
                            id="navbar-collapse",
                            navbar=True,
                        ),
                        width="auto",
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    sticky="top",
)


def render_page(page):
    """Render a full page layout with navbar."""
    return dbc.Container(
        [
            NAVBAR,
            html.Div(id="page-content", style={"marginTop": "20px"}),
            html.Footer(
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.P("Developed by Khaled Saifullah", className="text-white-50 text-center"),
                                    width=12,
                                ),
                            ],
                            className="mt-4",
                        ),
                    ],
                    className="bg-dark text-light py-3",
                ),
                style={"marginTop": "40px"},
            ),
        ],
        fluid=True,
    )