"""
Central configuration for the Financial Data Analytics & Forecasting Tool.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── API ──────────────────────────────────────────────────────────────────
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    # ── Dashboard ────────────────────────────────────────────────────────────
    DASH_DEBUG: bool = os.getenv("DASH_DEBUG", "False").lower() == "true"
    DASH_PORT: int = int(os.getenv("DASH_PORT", 8050))
    DASH_HOST: str = os.getenv("DASH_HOST", "127.0.0.1")

    # ── Data defaults ────────────────────────────────────────────────────────
    DEFAULT_TICKERS: list[str] = os.getenv(
        "DEFAULT_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA"
    ).split(",")
    DEFAULT_PERIOD: str = os.getenv("DEFAULT_PERIOD", "2y")
    VALID_PERIODS: list[str] = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

    # ── Forecasting ──────────────────────────────────────────────────────────
    FORECAST_HORIZON_DAYS: int = int(os.getenv("FORECAST_HORIZON_DAYS", 30))
    TEST_SIZE: float = 0.2          # 20% of data held out for evaluation
    RANDOM_STATE: int = 42

    # ── Pipeline ─────────────────────────────────────────────────────────────
    CACHE_DIR: str = "data/cache"
    REPORTS_DIR: str = "reports/output"
    MODELS_DIR: str = "models/saved"

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "logs/app.log"


config = Config()
