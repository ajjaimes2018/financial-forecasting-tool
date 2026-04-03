"""
Optimized data pipeline that orchestrates fetching → processing for
multiple tickers in a single pass.

Design goals:
- Process 500,000+ records efficiently using vectorised Pandas operations
- Cache intermediate results to avoid redundant I/O
- Return a combined DataFrame ready for modelling or dashboard rendering
"""

import time
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config
from data.fetcher import StockDataFetcher
from data.processor import StockDataProcessor


class DataPipeline:
    """
    End-to-end pipeline: fetch → process → combine.

    Uses a thread pool for concurrent downloads (I/O-bound),
    then sequentially applies CPU-bound feature engineering.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        period: str = config.DEFAULT_PERIOD,
        forecast_horizon: int = config.FORECAST_HORIZON_DAYS,
        max_workers: int = 5,
    ):
        self.tickers = [t.upper() for t in (tickers or config.DEFAULT_TICKERS)]
        self.period = period
        self.forecast_horizon = forecast_horizon
        self.max_workers = max_workers

        self._fetcher = StockDataFetcher()
        self._processor = StockDataProcessor(forecast_horizon=forecast_horizon)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline.

        Returns:
            Combined, feature-engineered DataFrame for all tickers.
            Includes a 'Ticker' column to identify each symbol.
        """
        start = time.perf_counter()
        logger.info(f"Pipeline starting for {len(self.tickers)} tickers: {self.tickers}")

        raw_data = self._fetch_concurrent()
        processed_frames = self._process_all(raw_data)
        combined = self._combine(processed_frames)

        elapsed = time.perf_counter() - start
        logger.info(
            f"Pipeline complete: {len(combined):,} rows across "
            f"{combined['Ticker'].nunique()} tickers in {elapsed:.2f}s"
        )
        return combined

    def run_single(self, ticker: str) -> pd.DataFrame | None:
        """Run the pipeline for a single ticker."""
        ticker = ticker.upper()
        raw = self._fetcher.fetch_single(ticker, self.period)
        if raw is None or raw.empty:
            logger.warning(f"No data for {ticker}")
            return None
        return self._processor.process(raw)

    def get_feature_columns(self) -> list[str]:
        return self._processor.get_feature_columns()

    # ── Internal steps ────────────────────────────────────────────────────────

    def _fetch_concurrent(self) -> dict[str, pd.DataFrame]:
        """Download all tickers concurrently using a thread pool."""
        results: dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetcher.fetch_single, ticker, self.period): ticker
                for ticker in self.tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[ticker] = df
                except Exception as exc:
                    logger.error(f"Fetch failed for {ticker}: {exc}")

        logger.info(f"Fetched data for {len(results)}/{len(self.tickers)} tickers")
        return results

    def _process_all(self, raw_data: dict[str, pd.DataFrame]) -> list[pd.DataFrame]:
        """Apply feature engineering to each ticker's raw DataFrame."""
        processed = []
        for ticker, df in raw_data.items():
            try:
                processed_df = self._processor.process(df)
                processed.append(processed_df)
                logger.debug(f"Processed {ticker}: {len(processed_df):,} rows")
            except Exception as exc:
                logger.error(f"Processing failed for {ticker}: {exc}")
        return processed

    def _combine(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate all processed DataFrames into one."""
        if not frames:
            raise ValueError("No data was successfully processed.")
        combined = pd.concat(frames, axis=0).sort_index()
        return combined


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = DataPipeline(tickers=["AAPL", "MSFT", "GOOGL"], period="2y")
    df = pipeline.run()
    print(df.groupby("Ticker").size().rename("rows"))
    print(f"\nTotal records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
