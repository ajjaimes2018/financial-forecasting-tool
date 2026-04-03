"""
Unit tests for the data pipeline.
Uses small synthetic DataFrames — no network calls required.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from data.processor import StockDataProcessor
from pipeline.data_pipeline import DataPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_ohlcv() -> pd.DataFrame:
    """Minimal synthetic OHLCV DataFrame (252 rows ≈ 1 trading year)."""
    n = 252
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 150 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "Open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "High":   close * (1 + rng.uniform(0, 0.01, n)),
        "Low":    close * (1 - rng.uniform(0, 0.01, n)),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        "Ticker": "TEST",
    }, index=dates)


# ── StockDataProcessor ────────────────────────────────────────────────────────

class TestStockDataProcessor:

    def test_output_contains_expected_features(self, raw_ohlcv):
        processor = StockDataProcessor(forecast_horizon=10)
        processed = processor.process(raw_ohlcv)
        for col in processor.get_feature_columns():
            assert col in processed.columns, f"Missing feature: {col}"

    def test_no_nulls_after_processing(self, raw_ohlcv):
        processor = StockDataProcessor(forecast_horizon=10)
        processed = processor.process(raw_ohlcv)
        assert processed.isnull().sum().sum() == 0

    def test_target_column_exists(self, raw_ohlcv):
        processor = StockDataProcessor(forecast_horizon=10)
        processed = processor.process(raw_ohlcv)
        assert "Future_Return" in processed.columns

    def test_row_count_reduced_by_lookback(self, raw_ohlcv):
        processor = StockDataProcessor(forecast_horizon=10)
        processed = processor.process(raw_ohlcv)
        assert len(processed) < len(raw_ohlcv)

    def test_rsi_within_bounds(self, raw_ohlcv):
        processor = StockDataProcessor(forecast_horizon=5)
        processed = processor.process(raw_ohlcv)
        assert processed["RSI_14"].between(0, 100).all()


# ── DataPipeline ──────────────────────────────────────────────────────────────

class TestDataPipeline:

    def test_run_returns_dataframe(self, raw_ohlcv):
        """Pipeline.run() should return a combined DataFrame."""
        pipeline = DataPipeline(tickers=["TEST"], period="1y", forecast_horizon=10)

        with patch.object(pipeline._fetcher, "fetch_single", return_value=raw_ohlcv):
            result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_ticker_column_preserved(self, raw_ohlcv):
        pipeline = DataPipeline(tickers=["TEST"], period="1y", forecast_horizon=10)
        with patch.object(pipeline._fetcher, "fetch_single", return_value=raw_ohlcv):
            result = pipeline.run()
        assert "Ticker" in result.columns
        assert "TEST" in result["Ticker"].values

    def test_empty_result_raises(self):
        pipeline = DataPipeline(tickers=["FAKE"], period="1y")
        with patch.object(pipeline._fetcher, "fetch_single", return_value=None):
            with pytest.raises(ValueError):
                pipeline.run()
