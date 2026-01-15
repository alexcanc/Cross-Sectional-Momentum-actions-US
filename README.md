# Cross-Sectional Momentum Strategy

A quantitative research project implementing a systematic momentum strategy on US equities, designed to demonstrate rigorous financial engineering methodology.

---

## Executive Summary

This project presents a fully-functional **cross-sectional momentum trading strategy** applied to the Dow Jones 30 index constituents. The implementation follows academic best practices established by Jegadeesh & Titman (1993) while incorporating modern regime-based risk management techniques.

### Key Features

- **Complete Backtesting Framework** - End-to-end pipeline from data ingestion to performance analytics
- **Interactive Dashboard** - Real-time strategy exploration via Streamlit
- **Machine Learning Integration** - Optional regime detection using interpretable ML models
- **Research-Grade Methodology** - Explicit handling of look-ahead bias, transaction costs, and other pitfalls

---

## Strategy Overview

| Component | Implementation |
|-----------|----------------|
| **Universe** | Dow Jones 30 constituents |
| **Signal** | 12-1 Momentum (12-month return, skip last month) |
| **Rebalancing** | Monthly |
| **Portfolio** | Long top 10 / Short bottom 10 (configurable) |
| **Weighting** | Equal-weight |
| **Costs** | 10 basis points per trade |
| **Regime Filter** | SMA-200 baseline / ML enhancement |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/Cross-Sectional-Momentum-actions-US.git
cd Cross-Sectional-Momentum-actions-US

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run app.py
```

The dashboard provides:
- Real-time parameter adjustment
- Performance visualization with benchmark comparison
- Current momentum rankings and portfolio positions
- Risk analytics including drawdown analysis
- Comprehensive methodology documentation

---

## Project Architecture

```
Cross-Sectional-Momentum-actions-US/
│
├── app.py                      # Streamlit dashboard (main entry point)
├── train_model.py              # ML model training script
├── run_walk_forward.py         # Walk-forward validation
│
├── src/                        # Core library
│   ├── config.py              # Centralized configuration
│   ├── data_loader.py         # Data acquisition & preprocessing
│   ├── signal.py              # Momentum signal computation
│   ├── backtest.py            # Backtesting engine
│   ├── metrics.py             # Performance analytics
│   ├── regime_filter.py       # SMA-based regime detection
│   ├── ml_regime.py           # ML regime classifier
│   └── walk_forward.py        # Time-series cross-validation
│
├── notebooks/                  # Research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_construction.ipynb
│   ├── 03_backtest.ipynb
│   ├── 04_regime_filter.ipynb
│   └── 05_ml_regime.ipynb
│
├── models/                     # Saved ML models
├── data/                       # Cached market data
└── report/                     # Research documentation
```

---

## Methodology

### 1. Momentum Signal Construction

The strategy employs the classical **12-1 momentum signal**, measuring the cumulative return over the past 12 months while excluding the most recent month:

```
Momentum(t) = Price(t-1) / Price(t-12) - 1
```

**Rationale for skipping the last month:** Short-term returns exhibit mean reversion (the "reversal effect"), which contaminates the medium-term momentum signal. This adjustment is standard in academic literature.

### 2. Portfolio Formation

At each monthly rebalancing date:

1. **Compute** momentum scores for all 30 stocks
2. **Rank** stocks cross-sectionally by momentum
3. **Long** the top N performers (winners)
4. **Short** the bottom N performers (losers) - optional
5. **Weight** positions equally within each leg
6. **Apply** transaction costs to turnover

### 3. Regime-Based Risk Management

#### Baseline: SMA-200 Filter

A trend-following overlay based on the S&P 500's position relative to its 200-day simple moving average:

| Market Condition | Signal | Exposure |
|------------------|--------|----------|
| Price > SMA-200 | Risk-On | 100% |
| Price < SMA-200 | Risk-Off | 50% |

#### Enhancement: ML Classifier

An optional logistic regression model trained on interpretable features:

- **Momentum indicators**: 1M, 3M, 6M returns
- **Volatility measures**: 20-day, 60-day realized volatility
- **Trend signals**: Price relative to SMA-50, SMA-200

The model is validated using expanding-window walk-forward analysis to prevent look-ahead bias.

---

## Performance Analytics

The framework computes industry-standard metrics:

| Category | Metrics |
|----------|---------|
| **Returns** | CAGR, Total Return, Annual Returns |
| **Risk** | Volatility, Max Drawdown, Drawdown Duration |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| **Operational** | Turnover, Transaction Costs, Hit Rate |

---

## Research Integrity

This project explicitly addresses common pitfalls in quantitative research:

| Concern | Mitigation |
|---------|------------|
| **Look-ahead bias** | Signals computed only with information available at decision time |
| **Data leakage** | Strict temporal separation between signal and execution |
| **Survivorship bias** | Acknowledged limitation; uses current index constituents |
| **Transaction costs** | Explicit modeling at 10 bps per trade |
| **Overfitting** | No parameter optimization; academic standard (12-1) used |
| **P-hacking** | Single pre-specified hypothesis tested |

---

## Machine Learning Module

### Training

```bash
# Standard training
python train_model.py

# With fresh data download
python train_model.py --fresh

# Custom training period
python train_model.py --train-end 2020-12-31
```

### Walk-Forward Validation

```bash
# Default configuration
python run_walk_forward.py

# Custom parameters
python run_walk_forward.py --train-years 5 --test-years 2 --step-years 1
```

---

## Programmatic Usage

```python
from src.data_loader import load_universe_data, load_benchmark_data
from src.signal import compute_momentum_signal
from src.backtest import run_backtest
from src.metrics import compute_metrics
from src.config import Config

# Initialize
config = Config()
prices = load_universe_data()
benchmark = load_benchmark_data()

# Execute strategy
results = run_backtest(prices, benchmark, config)

# Analyze performance
metrics = compute_metrics(
    results.strategy_returns,
    results.cumulative_returns,
    results.turnover,
    results.transaction_costs
)

print(f"CAGR: {metrics.cagr:.2%}")
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Max DD: {metrics.max_drawdown:.2%}")
```

---

## Known Limitations

1. **Survivorship Bias** - Uses current index constituents rather than point-in-time membership
2. **Universe Size** - 30 stocks limits diversification benefits
3. **Short Costs** - Borrowing fees and availability not modeled
4. **Market Impact** - Assumes perfect execution at monthly close prices
5. **Equal Weighting** - Sub-optimal but transparent allocation scheme

---

## Future Extensions

- Expand universe to S&P 100 or Russell 1000
- Implement sector-neutrality constraints
- Add alternative weighting schemes (risk parity, volatility targeting)
- Integrate fundamental quality screens
- Test international markets (MSCI World, Emerging Markets)

---

## References

- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency*. Journal of Finance.
- Carhart, M. M. (1997). *On Persistence in Mutual Fund Performance*. Journal of Finance.
- Moskowitz, T. J., & Grinblatt, M. (1999). *Do Industries Explain Momentum?*. Journal of Finance.
- Faber, M. (2007). *A Quantitative Approach to Tactical Asset Allocation*. Journal of Wealth Management.
- Asness, C., Moskowitz, T., & Pedersen, L. (2013). *Value and Momentum Everywhere*. Journal of Finance.

---

## Testing

The project includes a comprehensive test suite with **71 unit tests** covering all core modules.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_signal.py -v

# Run specific test class
pytest tests/test_metrics.py::TestComputeSharpeRatio -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `config.py` | 8 | Configuration validation |
| `signal.py` | 15 | Momentum signal & portfolio weights |
| `backtest.py` | 14 | Backtesting engine |
| `metrics.py` | 18 | Performance metrics |
| `regime_filter.py` | 16 | Market regime detection |

---

## Technical Requirements

- Python 3.9+
- Dependencies: pandas, numpy, scikit-learn, plotly, streamlit, yfinance, pytest

See `requirements.txt` for complete specification.

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

*This project is for educational and research purposes. Past performance does not guarantee future results. Not financial advice.*
