"""
Model evaluation: train/test split, metrics, and comparison utilities.
"""

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import config
from models.forecaster import BaseForecaster


class ModelEvaluator:
    """
    Evaluates one or more forecasters using time-series-aware
    train/test splits (no data leakage).
    """

    def __init__(self, test_size: float = config.TEST_SIZE):
        self.test_size = test_size

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        forecaster: BaseForecaster,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict:
        """
        Train on the first (1 - test_size) portion of data,
        evaluate on the remaining test_size portion.

        Returns a metrics dict.
        """
        train_df, test_df = self._temporal_split(df)
        logger.info(
            f"Evaluating {forecaster.MODEL_NAME}: "
            f"train={len(train_df):,}  test={len(test_df):,}"
        )

        forecaster.train(train_df, feature_columns)

        X_test = test_df[feature_columns]
        y_test = test_df["Future_Return"]
        y_pred = forecaster.pipeline.predict(X_test)

        metrics = self._compute_metrics(y_test.values, y_pred)
        metrics["model"] = forecaster.MODEL_NAME
        metrics["train_rows"] = len(train_df)
        metrics["test_rows"] = len(test_df)

        self._log_metrics(forecaster.MODEL_NAME, metrics)
        return metrics

    def compare(
        self,
        forecasters: list[BaseForecaster],
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        """
        Evaluate multiple forecasters and return a comparison DataFrame,
        sorted by R² descending.
        """
        rows = [self.evaluate(f, df, feature_columns) for f in forecasters]
        results = pd.DataFrame(rows).set_index("model").sort_values("r2", ascending=False)
        return results

    def cross_validate(
        self,
        forecaster: BaseForecaster,
        df: pd.DataFrame,
        feature_columns: list[str],
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """
        Time-series cross validation using sklearn TimeSeriesSplit.
        Returns per-fold metrics as a DataFrame.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = df[feature_columns].values
        y = df["Future_Return"].values

        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            forecaster.pipeline.fit(X[train_idx], y[train_idx])
            y_pred = forecaster.pipeline.predict(X[test_idx])
            m = self._compute_metrics(y[test_idx], y_pred)
            m["fold"] = fold
            fold_metrics.append(m)

        return pd.DataFrame(fold_metrics).set_index("fold")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _temporal_split(self, df: pd.DataFrame):
        """Split maintaining chronological order (no shuffle)."""
        split_idx = int(len(df) * (1 - self.test_size))
        return df.iloc[:split_idx], df.iloc[split_idx:]

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Directional accuracy: did we predict the right direction?
        direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

        return {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 4),
            "directional_accuracy": round(direction_acc, 4),
        }

    def _log_metrics(self, model_name: str, metrics: dict) -> None:
        logger.info(
            f"{model_name} → "
            f"R²={metrics['r2']:.4f} | "
            f"MAE={metrics['mae']:.6f} | "
            f"Dir.Acc={metrics['directional_accuracy']:.2%}"
        )
