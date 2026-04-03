"""
Dash layout definitions — header, sidebar, and main content panels.
Kept separate from callbacks for clarity.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

from config import config


# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "text": "#c9d1d9",
    "muted": "#8b949e",
}


def build_layout() -> html.Div:
    return html.Div(
        style={"backgroundColor": COLOURS["bg"], "minHeight": "100vh", "color": COLOURS["text"], "fontFamily": "monospace"},
        children=[
            _navbar(),
            dbc.Container(
                fluid=True,
                children=[
                    dbc.Row([
                        dbc.Col(_controls_panel(), md=3),
                        dbc.Col(_main_panel(), md=9),
                    ], className="mt-3"),
                ],
            ),
            # Hidden store for processed data
            dcc.Store(id="store-data"),
            dcc.Store(id="store-predictions"),
        ],
    )


# ── Sub-components ────────────────────────────────────────────────────────────

def _navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            html.Span("📈 Financial Analytics & Forecasting", style={"fontWeight": "bold", "fontSize": "1.2rem", "color": COLOURS["accent"]}),
            html.Span("Powered by yfinance · scikit-learn · Dash", style={"fontSize": "0.75rem", "color": COLOURS["muted"]}),
        ]),
        color=COLOURS["surface"],
        dark=True,
        style={"borderBottom": f"1px solid {COLOURS['border']}"},
    )


def _controls_panel() -> dbc.Card:
    return dbc.Card(
        style={"backgroundColor": COLOURS["surface"], "border": f"1px solid {COLOURS['border']}"},
        children=dbc.CardBody([
            html.H6("Configuration", style={"color": COLOURS["accent"]}),
            html.Hr(style={"borderColor": COLOURS["border"]}),

            html.Label("Tickers", style={"fontSize": "0.8rem", "color": COLOURS["muted"]}),
            dcc.Dropdown(
                id="input-tickers",
                options=[{"label": t, "value": t} for t in config.DEFAULT_TICKERS],
                value=config.DEFAULT_TICKERS[:3],
                multi=True,
                style={"backgroundColor": COLOURS["bg"], "color": "#fff"},
            ),

            html.Br(),
            html.Label("Period", style={"fontSize": "0.8rem", "color": COLOURS["muted"]}),
            dcc.Dropdown(
                id="input-period",
                options=[{"label": p, "value": p} for p in config.VALID_PERIODS],
                value=config.DEFAULT_PERIOD,
                clearable=False,
                style={"backgroundColor": COLOURS["bg"]},
            ),

            html.Br(),
            html.Label("Forecast Horizon (days)", style={"fontSize": "0.8rem", "color": COLOURS["muted"]}),
            dcc.Slider(
                id="input-horizon",
                min=5, max=60, step=5,
                value=config.FORECAST_HORIZON_DAYS,
                marks={v: str(v) for v in [5, 15, 30, 45, 60]},
            ),

            html.Br(),
            html.Label("Model", style={"fontSize": "0.8rem", "color": COLOURS["muted"]}),
            dcc.RadioItems(
                id="input-model",
                options=[
                    {"label": " Random Forest", "value": "random_forest"},
                    {"label": " Gradient Boosting", "value": "gradient_boosting"},
                    {"label": " Ridge (baseline)", "value": "linear_regression"},
                ],
                value="random_forest",
                labelStyle={"display": "block", "marginBottom": "4px", "fontSize": "0.85rem"},
            ),

            html.Br(),
            dbc.Button(
                "▶  Run Pipeline",
                id="btn-run",
                color="primary",
                className="w-100",
                style={"backgroundColor": COLOURS["accent"], "borderColor": COLOURS["accent"]},
            ),

            html.Br(),
            dbc.Button(
                "📥  Export Report",
                id="btn-export",
                outline=True,
                color="secondary",
                className="w-100 mt-2",
            ),
            html.Div(id="export-status", style={"fontSize": "0.75rem", "color": COLOURS["green"], "marginTop": "6px"}),
        ]),
    )


def _main_panel() -> html.Div:
    return html.Div([
        # KPI cards
        dbc.Row(id="kpi-row", className="mb-3"),

        # Price chart
        dbc.Card(
            style={"backgroundColor": COLOURS["surface"], "border": f"1px solid {COLOURS['border']}"},
            children=dbc.CardBody([
                html.H6("Price History", style={"color": COLOURS["accent"]}),
                dcc.Loading(dcc.Graph(id="chart-price", style={"height": "320px"})),
            ]),
        ),

        html.Br(),

        dbc.Row([
            # Returns distribution
            dbc.Col(dbc.Card(
                style={"backgroundColor": COLOURS["surface"], "border": f"1px solid {COLOURS['border']}"},
                children=dbc.CardBody([
                    html.H6("Returns Distribution", style={"color": COLOURS["accent"]}),
                    dcc.Loading(dcc.Graph(id="chart-returns", style={"height": "260px"})),
                ]),
            ), md=6),

            # Feature importance
            dbc.Col(dbc.Card(
                style={"backgroundColor": COLOURS["surface"], "border": f"1px solid {COLOURS['border']}"},
                children=dbc.CardBody([
                    html.H6("Feature Importance", style={"color": COLOURS["accent"]}),
                    dcc.Loading(dcc.Graph(id="chart-importance", style={"height": "260px"})),
                ]),
            ), md=6),
        ]),

        html.Br(),

        # Predictions table
        dbc.Card(
            style={"backgroundColor": COLOURS["surface"], "border": f"1px solid {COLOURS['border']}"},
            children=dbc.CardBody([
                html.H6("Model Predictions", style={"color": COLOURS["accent"]}),
                html.Div(id="table-predictions"),
            ]),
        ),
    ])
