"""
Feature engineering and preprocessing for stock market data.
Adds technical indicators and prepares data for ML models.
"""

import pandas as pd
import numpy as np
from loguru import logger


class StockDataProcessor:
    """
    Transforms raw OHLCV DataFrames into feature-rich datasets
    suitable for training and forecasting.

    Features added:
    - Returns (daily, 5-day, 20-day)
    - Moving averages (SMA 20, SMA 50, EMA 12, EMA 26)
    - Volatility (rolling std of returns)
    - RSI (14-period)
    - MACD & Signal line
    - Bollinger Bands
    - Volume change
    """

    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full processing pipeline: add features → add target → drop NaNs.

        Args:
            df: Raw OHLCV DataFrame from StockDataFetcher

        Returns:
            Feature-engineered DataFrame with target column 'Future_Return'
        """
        df = df.copy()
        df = self._add_return_features(df)
        df = self._add_moving_averages(df)
        df = self._add_volatility(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_volume_features(df)
        df = self._add_target(df)
        df.dropna(inplace=True)
        logger.debug(f"Processed DataFrame: {len(df):,} rows, {len(df.columns)} features")
        return df

    def get_feature_columns(self) -> list[str]:
        """Return the list of feature column names used for modelling."""
        return [
            "Return_1d", "Return_5d", "Return_20d",
            "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "Volatility_20",
            "RSI_14",
            "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Lower", "BB_Width",
            "Volume_Change", "Volume_SMA_20",
        ]

    # ── Feature builders ──────────────────────────────────────────────────────

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Return_1d"] = df["Close"].pct_change(1)
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_20d"] = df["Close"].pct_change(20)
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Volatility_20"] = df["Return_1d"].rolling(20).std()
        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        sma = df["Close"].rolling(period).mean()
        std = df["Close"].rolling(period).std()
        df["BB_Upper"] = sma + 2 * std
        df["BB_Lower"] = sma - 2 * std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: forward return over forecast_horizon days.
        Positive = price went up (buy signal).
        """
        future_close = df["Close"].shift(-self.forecast_horizon)
        df["Future_Return"] = (future_close - df["Close"]) / df["Close"]
        return df


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.fetcher import StockDataFetcher

    fetcher = StockDataFetcher()
    raw = fetcher.fetch_single("AAPL", period="2y")
    processor = StockDataProcessor(forecast_horizon=30)
    processed = processor.process(raw)
    print(processed[processor.get_feature_columns()].tail(5))
    print(f"\nShape: {processed.shape}")
