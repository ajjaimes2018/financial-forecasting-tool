"""
Entry point for the Financial Analytics & Forecasting Tool.

Usage:
    python main.py                  # Launch the Dash dashboard
    python main.py --cli            # Run pipeline in CLI mode (no browser)
    python main.py --export         # Run pipeline + export Excel report
"""

import argparse
import os
import sys
from loguru import logger

from config import config


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(config.LOG_FILE, level="DEBUG", rotation="10 MB", retention="7 days")


def run_dashboard():
    from dashboard.app import create_app
    app = create_app()
    logger.info(f"Dashboard running at http://{config.DASH_HOST}:{config.DASH_PORT}")
    app.run(host=config.DASH_HOST, port=config.DASH_PORT, debug=config.DASH_DEBUG)


def run_cli(export: bool = False):
    from pipeline.data_pipeline import DataPipeline
    from models.forecaster import RandomForestForecaster
    from models.evaluator import ModelEvaluator
    from reports.generator import ReportGenerator

    logger.info("Starting CLI pipeline…")
    pipeline = DataPipeline()
    df = pipeline.run()

    forecaster = RandomForestForecaster()
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(forecaster, df, pipeline.get_feature_columns())

    predictions = forecaster.predict_latest(df)

    logger.info("\n── Predictions ──────────────────────────────")
    for p in predictions:
        logger.info(f"  {p['Ticker']:6}  {p['Predicted_Return']:+.2%}  [{p['Signal']}]")

    logger.info(f"\n── Model Metrics ({metrics['model']}) ──────────────")
    logger.info(f"  R²:                  {metrics['r2']:.4f}")
    logger.info(f"  Directional Acc:     {metrics['directional_accuracy']:.1%}")
    logger.info(f"  MAE:                 {metrics['mae']:.6f}")
    logger.info(f"  RMSE:                {metrics['rmse']:.6f}")

    if export:
        reporter = ReportGenerator()
        path = reporter.generate_excel_report(df, predictions)
        logger.info(f"\n  Report exported → {path}")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Financial Analytics & Forecasting Tool")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no browser)")
    parser.add_argument("--export", action="store_true", help="Export Excel report after pipeline run")
    args = parser.parse_args()

    if args.cli or args.export:
        run_cli(export=args.export)
    else:
        run_dashboard()
