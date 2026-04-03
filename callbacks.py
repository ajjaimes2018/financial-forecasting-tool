"""
Dash callbacks: wire up user interactions to data pipeline and charts.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback, html, dash_table
import dash_bootstrap_components as dbc
from loguru import logger

from config import config
from pipeline.data_pipeline import DataPipeline
from models.forecaster import RandomForestForecaster, GradientBoostingForecaster, LinearRegressionForecaster
from models.evaluator import ModelEvaluator
from reports.generator import ReportGenerator
from dashboard.layouts import COLOURS


MODEL_MAP = {
    "random_forest": RandomForestForecaster,
    "gradient_boosting": GradientBoostingForecaster,
    "linear_regression": LinearRegressionForecaster,
}


def register_callbacks(app):
    """Register all Dash callbacks with the app instance."""

    # ── Run Pipeline ─────────────────────────────────────────────────────────

    @app.callback(
        Output("store-data", "data"),
        Output("store-predictions", "data"),
        Output("kpi-row", "children"),
        Input("btn-run", "n_clicks"),
        State("input-tickers", "value"),
        State("input-period", "value"),
        State("input-horizon", "value"),
        State("input-model", "value"),
        prevent_initial_call=True,
    )
    def run_pipeline(n_clicks, tickers, period, horizon, model_name):
        if not tickers:
            return {}, {}, []

        try:
            pipeline = DataPipeline(tickers=tickers, period=period, forecast_horizon=int(horizon))
            df = pipeline.run()

            forecaster_cls = MODEL_MAP[model_name]
            forecaster = forecaster_cls()
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(forecaster, df, pipeline.get_feature_columns())

            predictions = forecaster.predict_latest(df)

            kpi_cards = _build_kpi_cards(metrics, len(df))

            return (
                df.reset_index().to_json(date_format="iso"),
                json.dumps(predictions),
                kpi_cards,
            )

        except Exception as exc:
            logger.error(f"Pipeline error: {exc}")
            return {}, {}, [dbc.Col(html.Div(f"Error: {exc}", style={"color": COLOURS["red"]}))]

    # ── Price Chart ──────────────────────────────────────────────────────────

    @app.callback(
        Output("chart-price", "figure"),
        Input("store-data", "data"),
        prevent_initial_call=True,
    )
    def update_price_chart(data):
        if not data:
            return go.Figure()
        df = pd.read_json(data)
        df["Date"] = pd.to_datetime(df["Date"])

        fig = go.Figure()
        for ticker in df["Ticker"].unique():
            subset = df[df["Ticker"] == ticker].sort_values("Date")
            fig.add_trace(go.Scatter(
                x=subset["Date"], y=subset["Close"],
                mode="lines", name=ticker, line={"width": 1.5},
            ))

        fig.update_layout(**_dark_layout("Closing Price"))
        return fig

    # ── Returns Distribution ─────────────────────────────────────────────────

    @app.callback(
        Output("chart-returns", "figure"),
        Input("store-data", "data"),
        prevent_initial_call=True,
    )
    def update_returns_chart(data):
        if not data:
            return go.Figure()
        df = pd.read_json(data)

        fig = go.Figure()
        for ticker in df["Ticker"].unique():
            subset = df[df["Ticker"] == ticker]
            fig.add_trace(go.Histogram(
                x=subset["Return_1d"],
                name=ticker,
                opacity=0.7,
                nbinsx=50,
            ))

        fig.update_layout(**_dark_layout("Daily Return Distribution"), barmode="overlay")
        return fig

    # ── Feature Importance ───────────────────────────────────────────────────

    @app.callback(
        Output("chart-importance", "figure"),
        Input("store-data", "data"),
        State("input-model", "value"),
        State("input-tickers", "value"),
        State("input-period", "value"),
        State("input-horizon", "value"),
        prevent_initial_call=True,
    )
    def update_importance_chart(data, model_name, tickers, period, horizon):
        if not data or model_name != "random_forest":
            fig = go.Figure()
            fig.update_layout(**_dark_layout("Feature Importance (Random Forest only)"))
            return fig

        try:
            df = pd.read_json(data)
            pipeline = DataPipeline(tickers=tickers or config.DEFAULT_TICKERS, period=period, forecast_horizon=int(horizon))
            forecaster = RandomForestForecaster()
            forecaster.train(df, pipeline.get_feature_columns())
            importances = forecaster.feature_importances().head(10)

            fig = go.Figure(go.Bar(
                x=importances.values,
                y=importances.index,
                orientation="h",
                marker_color=COLOURS["accent"],
            ))
            fig.update_layout(**_dark_layout("Top 10 Feature Importances"))
        except Exception as exc:
            logger.error(f"Importance chart error: {exc}")
            fig = go.Figure()

        return fig

    # ── Predictions Table ────────────────────────────────────────────────────

    @app.callback(
        Output("table-predictions", "children"),
        Input("store-predictions", "data"),
        prevent_initial_call=True,
    )
    def update_predictions_table(data):
        if not data:
            return html.P("Run the pipeline to see predictions.", style={"color": COLOURS["muted"]})

        predictions = json.loads(data)
        if not predictions:
            return html.P("No predictions available.", style={"color": COLOURS["muted"]})

        rows = []
        for p in predictions:
            signal = p.get("Signal", "N/A")
            colour = COLOURS["green"] if signal == "BUY" else COLOURS["red"] if signal == "SELL" else COLOURS["muted"]
            rows.append(
                html.Tr([
                    html.Td(p.get("Ticker", "")),
                    html.Td(f"{p.get('Predicted_Return', 0):.2%}"),
                    html.Td(signal, style={"color": colour, "fontWeight": "bold"}),
                ])
            )

        return dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Ticker"), html.Th("Predicted Return"), html.Th("Signal")])),
                html.Tbody(rows),
            ],
            bordered=False,
            striped=True,
            hover=True,
            style={"color": COLOURS["text"], "fontSize": "0.9rem"},
        )

    # ── Export Report ────────────────────────────────────────────────────────

    @app.callback(
        Output("export-status", "children"),
        Input("btn-export", "n_clicks"),
        State("store-data", "data"),
        State("store-predictions", "data"),
        prevent_initial_call=True,
    )
    def export_report(n_clicks, data, predictions_data):
        if not data:
            return "⚠ Run the pipeline first."
        try:
            df = pd.read_json(data)
            predictions = json.loads(predictions_data) if predictions_data else None
            reporter = ReportGenerator()
            path = reporter.generate_excel_report(df, predictions)
            return f"✓ Saved to {path}"
        except Exception as exc:
            return f"✗ Export failed: {exc}"


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _dark_layout(title: str) -> dict:
    return {
        "title": {"text": title, "font": {"color": COLOURS["text"], "size": 13}},
        "plot_bgcolor": COLOURS["bg"],
        "paper_bgcolor": COLOURS["surface"],
        "font": {"color": COLOURS["text"]},
        "xaxis": {"gridcolor": COLOURS["border"], "showgrid": True},
        "yaxis": {"gridcolor": COLOURS["border"], "showgrid": True},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    }


def _build_kpi_cards(metrics: dict, total_rows: int) -> list:
    cards_data = [
        ("Records Processed", f"{total_rows:,}", COLOURS["accent"]),
        ("Directional Accuracy", f"{metrics['directional_accuracy']:.1%}", COLOURS["green"]),
        ("R² Score", f"{metrics['r2']:.4f}", COLOURS["green"] if metrics["r2"] > 0 else COLOURS["red"]),
        ("RMSE", f"{metrics['rmse']:.6f}", COLOURS["muted"]),
    ]
    cols = []
    for label, value, colour in cards_data:
        cols.append(dbc.Col(dbc.Card(
            style={"backgroundColor": COLOURS["surface"], "border": f"1px solid {COLOURS['border']}"},
            children=dbc.CardBody([
                html.P(label, style={"fontSize": "0.75rem", "color": COLOURS["muted"], "marginBottom": "2px"}),
                html.H5(value, style={"color": colour, "marginBottom": 0}),
            ], style={"padding": "12px"}),
        ), md=3))
    return cols
