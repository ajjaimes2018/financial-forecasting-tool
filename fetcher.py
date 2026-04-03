"""
Stock data fetcher using Yahoo Finance (yfinance).
Handles downloading, caching, and basic validation of OHLCV data.
"""

import os
import hashlib
import pandas as pd
import yfinance as yf
from loguru import logger
from pathlib import Path

from config import config


class StockDataFetcher:
    """
    Downloads historical OHLCV data for one or more tickers.
    Results are cached to disk to avoid redundant API calls.
    """

    def __init__(self, cache_dir: str = config.CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, tickers: list[str], period: str = config.DEFAULT_PERIOD) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for a list of tickers.

        Args:
            tickers: List of stock symbols, e.g. ["AAPL", "MSFT"]
            period:  yfinance period string, e.g. "2y", "6mo"

        Returns:
            Dict mapping ticker → cleaned DataFrame with columns:
            [Open, High, Low, Close, Volume, Ticker]
        """
        results: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            ticker = ticker.upper().strip()
            try:
                df = self._load_or_download(ticker, period)
                if df is not None and not df.empty:
                    results[ticker] = df
                    logger.info(f"Fetched {len(df):,} rows for {ticker}")
                else:
                    logger.warning(f"No data returned for {ticker}")
            except Exception as exc:
                logger.error(f"Failed to fetch {ticker}: {exc}")

        return results

    def fetch_single(self, ticker: str, period: str = config.DEFAULT_PERIOD) -> pd.DataFrame | None:
        """Convenience wrapper to fetch a single ticker."""
        result = self.fetch([ticker], period)
        return result.get(ticker.upper())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cache_path(self, ticker: str, period: str) -> Path:
        key = hashlib.md5(f"{ticker}-{period}".encode()).hexdigest()[:8]
        return self.cache_dir / f"{ticker}_{period}_{key}.parquet"

    def _load_or_download(self, ticker: str, period: str) -> pd.DataFrame | None:
        cache_file = self._cache_path(ticker, period)

        if cache_file.exists():
            logger.debug(f"Loading {ticker} from cache: {cache_file.name}")
            return pd.read_parquet(cache_file)

        logger.debug(f"Downloading {ticker} ({period}) from Yahoo Finance…")
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

        if raw.empty:
            return None

        df = self._clean(raw, ticker)
        df.to_parquet(cache_file, index=True)
        return df

    def _clean(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardise column names, drop nulls, add Ticker column."""
        df = raw.copy()

        # yfinance may return MultiIndex columns when downloading multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df["Ticker"] = ticker
        return df


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    data = fetcher.fetch(["AAPL", "MSFT"], period="1y")
    for sym, df in data.items():
        print(f"\n{sym}: {len(df):,} rows | {df.index[0].date()} → {df.index[-1].date()}")
        print(df.tail(3))
