"""
Unit tests for forecasting models and evaluator.
"""

import pytest
import numpy as np
import pandas as pd

from models.forecaster import RandomForestForecaster, LinearRegressionForecaster
from models.evaluator import ModelEvaluator


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic processed DataFrame for model testing."""
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    data = {col: rng.standard_normal(n) for col in [
        "Return_1d", "Return_5d", "Return_20d",
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "Volatility_20", "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Lower", "BB_Width",
        "Volume_Change", "Volume_SMA_20",
        "Future_Return",
    ]}
    data["RSI_14"] = np.clip(data["RSI_14"] * 15 + 50, 0, 100)
    data["Ticker"] = "TEST"
    return pd.DataFrame(data, index=dates)


FEATURES = [
    "Return_1d", "Return_5d", "Return_20d",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "Volatility_20", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Width",
    "Volume_Change", "Volume_SMA_20",
]


# ── RandomForestForecaster ────────────────────────────────────────────────────

class TestRandomForestForecaster:

    def test_train_and_predict(self, feature_df):
        model = RandomForestForecaster(n_estimators=10)
        model.train(feature_df, FEATURES)
        preds = model.predict(feature_df)
        assert len(preds) == len(feature_df)
        assert not np.any(np.isnan(preds))

    def test_raises_before_training(self, feature_df):
        model = RandomForestForecaster()
        with pytest.raises(RuntimeError):
            model.predict(feature_df)

    def test_feature_importances_sum_to_one(self, feature_df):
        model = RandomForestForecaster(n_estimators=10)
        model.train(feature_df, FEATURES)
        importances = model.feature_importances()
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_predict_latest_returns_list(self, feature_df):
        model = RandomForestForecaster(n_estimators=10)
        model.train(feature_df, FEATURES)
        result = model.predict_latest(feature_df)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "Signal" in result[0]


# ── ModelEvaluator ────────────────────────────────────────────────────────────

class TestModelEvaluator:

    def test_evaluate_returns_metrics(self, feature_df):
        model = LinearRegressionForecaster()
        evaluator = ModelEvaluator(test_size=0.2)
        metrics = evaluator.evaluate(model, feature_df, FEATURES)

        for key in ["mae", "rmse", "r2", "directional_accuracy"]:
            assert key in metrics

    def test_directional_accuracy_in_range(self, feature_df):
        model = LinearRegressionForecaster()
        evaluator = ModelEvaluator(test_size=0.2)
        metrics = evaluator.evaluate(model, feature_df, FEATURES)
        assert 0.0 <= metrics["directional_accuracy"] <= 1.0

    def test_compare_returns_dataframe(self, feature_df):
        models = [LinearRegressionForecaster(), RandomForestForecaster(n_estimators=10)]
        evaluator = ModelEvaluator(test_size=0.2)
        results = evaluator.compare(models, feature_df, FEATURES)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
