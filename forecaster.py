"""
Predictive models for stock trend forecasting.

Models available:
- RandomForestForecaster  (default, best balance of accuracy/speed)
- GradientBoostingForecaster
- LinearRegressionForecaster (baseline)

Each model:
 1. Trains on historical feature data
 2. Predicts forward return over `forecast_horizon` days
 3. Can be saved/loaded for reuse
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import config


class BaseForecaster:
    """Abstract base for all forecasting models."""

    MODEL_NAME = "base"

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []
        self._is_trained = False

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, feature_columns: list[str]) -> None:
        """
        Fit the model on processed historical data.

        Args:
            df:              Feature-engineered DataFrame (output of DataPipeline)
            feature_columns: Column names to use as model inputs
        """
        self.feature_columns = feature_columns
        X, y = self._split_xy(df)
        logger.info(f"Training {self.MODEL_NAME} on {len(X):,} samples…")
        self.pipeline.fit(X, y)
        self._is_trained = True
        logger.info(f"{self.MODEL_NAME} training complete")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict future returns for rows in df."""
        self._check_trained()
        X = df[self.feature_columns]
        return self.pipeline.predict(X)

    def predict_latest(self, df: pd.DataFrame) -> dict:
        """
        Predict the expected return for the most recent date in df.

        Returns a dict with ticker, date, predicted_return, and signal.
        """
        self._check_trained()
        latest = df.sort_index().groupby("Ticker").tail(1)
        X = latest[self.feature_columns]
        preds = self.pipeline.predict(X)

        results = []
        for (ticker, row), pred in zip(latest.iterrows(), preds):
            signal = "BUY" if pred > 0.02 else "SELL" if pred < -0.02 else "HOLD"
            results.append({
                "Ticker": latest.loc[row.name, "Ticker"] if "Ticker" in latest.columns else "N/A",
                "Date": row.name if hasattr(row, "name") else None,
                "Predicted_Return": round(float(pred), 4),
                "Signal": signal,
            })
        return results

    def save(self, directory: str = config.MODELS_DIR) -> str:
        """Persist the trained pipeline to disk."""
        self._check_trained()
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = os.path.join(directory, f"{self.MODEL_NAME}.joblib")
        joblib.dump({"pipeline": self.pipeline, "features": self.feature_columns}, path)
        logger.info(f"Model saved → {path}")
        return path

    def load(self, directory: str = config.MODELS_DIR) -> None:
        """Load a previously saved pipeline from disk."""
        path = os.path.join(directory, f"{self.MODEL_NAME}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}")
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.feature_columns = data["features"]
        self._is_trained = True
        logger.info(f"Model loaded ← {path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split_xy(self, df: pd.DataFrame):
        X = df[self.feature_columns]
        y = df["Future_Return"]
        return X, y

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError(f"{self.MODEL_NAME} has not been trained yet. Call .train() first.")


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest regressor — robust, handles non-linearity well,
    and provides feature importances for explainability.
    """
    MODEL_NAME = "random_forest"

    def __init__(self, n_estimators: int = 200, max_depth: int = 8):
        super().__init__()
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    def feature_importances(self) -> pd.Series:
        self._check_trained()
        model = self.pipeline.named_steps["model"]
        return pd.Series(
            model.feature_importances_,
            index=self.feature_columns,
        ).sort_values(ascending=False)


class GradientBoostingForecaster(BaseForecaster):
    """
    Gradient Boosting regressor — often more accurate than RF
    but slower to train.
    """
    MODEL_NAME = "gradient_boosting"

    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05):
        super().__init__()
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=4,
                random_state=config.RANDOM_STATE,
            )),
        ])


class LinearRegressionForecaster(BaseForecaster):
    """
    Ridge regression baseline — fast and interpretable.
    Useful for benchmarking against tree-based models.
    """
    MODEL_NAME = "linear_regression"

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
