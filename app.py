"""
Dash application factory.
Creates and configures the app instance.
"""

import dash
import dash_bootstrap_components as dbc

from dashboard.layouts import build_layout
from dashboard.callbacks import register_callbacks


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Financial Analytics & Forecasting",
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    app.layout = build_layout()
    register_callbacks(app)
    return app
