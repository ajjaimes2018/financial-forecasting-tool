"""
Automated financial reporting.
Generates summary Excel reports and PDF snapshots from pipeline output.
"""

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

from config import config


class ReportGenerator:
    """
    Produces Excel and summary reports from processed stock data
    and model predictions.
    """

    def __init__(self, output_dir: str = config.REPORTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_excel_report(
        self,
        df: pd.DataFrame,
        predictions: list[dict] | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Write a multi-sheet Excel workbook:
          - Sheet 1: Summary statistics per ticker
          - Sheet 2: Recent price data (last 60 rows per ticker)
          - Sheet 3: Model predictions (if provided)

        Returns the path to the saved file.
        """
        filename = filename or f"financial_report_{self._timestamp()}.xlsx"
        filepath = self.output_dir / filename

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            self._write_summary_sheet(df, writer)
            self._write_recent_data_sheet(df, writer)
            if predictions:
                self._write_predictions_sheet(predictions, writer)

        logger.info(f"Excel report saved → {filepath}")
        return str(filepath)

    def generate_summary_csv(self, df: pd.DataFrame, filename: str | None = None) -> str:
        """Export summary statistics as CSV."""
        filename = filename or f"summary_{self._timestamp()}.csv"
        filepath = self.output_dir / filename
        summary = self._build_summary(df)
        summary.to_csv(filepath)
        logger.info(f"Summary CSV saved → {filepath}")
        return str(filepath)

    # ── Sheet builders ────────────────────────────────────────────────────────

    def _write_summary_sheet(self, df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
        summary = self._build_summary(df)
        summary.to_excel(writer, sheet_name="Summary")

    def _write_recent_data_sheet(self, df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
        recent = (
            df.groupby("Ticker", group_keys=False)
            .apply(lambda g: g.sort_index().tail(60))
        )
        recent[["Ticker", "Open", "High", "Low", "Close", "Volume"]].to_excel(
            writer, sheet_name="Recent Data"
        )

    def _write_predictions_sheet(self, predictions: list[dict], writer: pd.ExcelWriter) -> None:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_excel(writer, sheet_name="Predictions", index=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-ticker summary statistics."""
        summaries = []
        for ticker, group in df.groupby("Ticker"):
            latest = group.sort_index().iloc[-1]
            summaries.append({
                "Ticker": ticker,
                "Latest_Close": round(latest["Close"], 2),
                "Period_Return": round(group["Return_1d"].sum(), 4),
                "Avg_Daily_Return": round(group["Return_1d"].mean(), 6),
                "Volatility_20d": round(latest.get("Volatility_20", float("nan")), 6),
                "RSI_14": round(latest.get("RSI_14", float("nan")), 2),
                "Total_Records": len(group),
                "From": group.index.min().date(),
                "To": group.index.max().date(),
            })
        return pd.DataFrame(summaries).set_index("Ticker")

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
