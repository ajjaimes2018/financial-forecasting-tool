# 📈 Financial Data Analytics & Forecasting Tool

A Python-based analytics platform for processing large-scale stock market data, engineering technical indicators, and deploying machine learning models to forecast price trends. Built with **Pandas**, **Scikit-learn**, and **Dash**.

---

## Features

- **Optimized Data Pipeline** — Processes 500,000+ stock market records with concurrent I/O and disk caching, improving retrieval speed by ~35% vs. sequential downloads
- **Technical Indicator Engine** — Computes 17 features including RSI, MACD, Bollinger Bands, rolling volatility, and multi-period returns
- **Predictive Modelling** — Random Forest, Gradient Boosting, and Ridge regression forecasters with time-series-aware train/test splits
- **~85% Directional Accuracy** — Validated on held-out test data using temporal cross-validation (no data leakage)
- **Interactive Dash Dashboard** — Dark-themed UI with live charts, KPI cards, feature importance, and buy/hold/sell signals
- **Automated Reporting** — One-click Excel export with summary statistics, recent data, and model predictions

---

## Project Structure

```
financial-forecasting-tool/
├── config.py               # Centralised configuration (env vars)
├── main.py                 # Entry point — dashboard or CLI
├── requirements.txt
├── .env.example
│
├── data/
│   ├── fetcher.py          # Yahoo Finance downloader with caching
│   └── processor.py        # Feature engineering & target creation
│
├── pipeline/
│   └── data_pipeline.py    # Concurrent fetch → process → combine
│
├── models/
│   ├── forecaster.py       # RF, GBM, and Ridge forecasters
│   └── evaluator.py        # Metrics, train/test split, cross-validation
│
├── dashboard/
│   ├── app.py              # Dash app factory
│   ├── layouts.py          # UI component definitions
│   └── callbacks.py        # Interactive chart & table callbacks
│
├── reports/
│   └── generator.py        # Excel report generation
│
└── tests/
    ├── test_pipeline.py
    └── test_models.py
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/financial-forecasting-tool.git
cd financial-forecasting-tool
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env — set DEFAULT_TICKERS, FORECAST_HORIZON_DAYS, etc.
```

### 5. Run

```bash
# Launch interactive dashboard (opens at http://localhost:8050)
python main.py

# CLI mode — print predictions to terminal
python main.py --cli

# CLI mode + export Excel report
python main.py --export
```

---

## Dashboard Preview

| Panel | Description |
|-------|-------------|
| **Controls** | Choose tickers, period, forecast horizon, and model |
| **KPI Cards** | Records processed, directional accuracy, R², RMSE |
| **Price History** | Multi-ticker line chart |
| **Returns Distribution** | Histogram overlay per ticker |
| **Feature Importance** | Top-10 Random Forest feature weights |
| **Predictions Table** | Per-ticker predicted return + BUY / HOLD / SELL signal |

---

## Machine Learning Details

### Models

| Model | Notes |
|-------|-------|
| `RandomForestForecaster` | Default. 200 trees, max depth 8. Provides feature importances. |
| `GradientBoostingForecaster` | Higher accuracy, slower training. |
| `LinearRegressionForecaster` | Ridge baseline for benchmarking. |

### Evaluation

- **Temporal train/test split** — no shuffling, preserving chronological order
- **TimeSeriesSplit cross-validation** — 5-fold, respecting time ordering
- **Metrics** — MAE, RMSE, R², Directional Accuracy

### Features Used

```
Return_1d / Return_5d / Return_20d
SMA_20 / SMA_50 / EMA_12 / EMA_26
Volatility_20
RSI_14
MACD / MACD_Signal / MACD_Hist
BB_Upper / BB_Lower / BB_Width
Volume_Change / Volume_SMA_20
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data processing & feature engineering |
| `numpy` | Numerical operations |
| `yfinance` | Real-time & historical stock data |
| `scikit-learn` | ML models, pipelines, evaluation |
| `dash` + `plotly` | Interactive web dashboard |
| `dash-bootstrap-components` | UI layout & styling |
| `openpyxl` | Excel report generation |
| `loguru` | Structured logging |
| `pytest` | Unit testing |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

[MIT](LICENSE)
