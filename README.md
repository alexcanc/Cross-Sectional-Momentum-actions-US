# Cross-Sectional Momentum Strategy - US Equities

## Overview

This project implements a **Cross-Sectional Momentum strategy** on US equities (Dow Jones 30), following academic standards and quantitative research best practices.

**Target audience**: ESCP / Polytechnique (X-HEC) admission committees
**Objective**: Demonstrate rigorous quantitative research methodology, not over-optimized returns

## Strategy Summary

| Parameter | Value |
|-----------|-------|
| Universe | Dow Jones 30 |
| Signal | 12-1 Momentum (12-month return minus last month) |
| Rebalancing | Monthly |
| Portfolio | Long top 10 / Short bottom 10 (or Long-only variant) |
| Weighting | Equal-weight |
| Transaction costs | 10 bps per trade |

## Project Structure

```
Cross-Sectional-Momentum-actions-US/
├── data/                          # Cached price data (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data download & preprocessing
│   ├── signal.py                 # Momentum signal computation
│   ├── backtest.py               # Backtesting engine
│   ├── metrics.py                # Performance metrics
│   ├── regime_filter.py          # Market regime detection
│   └── config.py                 # Configuration parameters
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_construction.ipynb
│   ├── 03_backtest.ipynb
│   └── 04_regime_filter.ipynb
├── report/
│   └── research_report.ipynb     # Final research paper (PDF-exportable)
├── app.py                        # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Cross-Sectional-Momentum-actions-US.git
cd Cross-Sectional-Momentum-actions-US

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard allows you to:
- Adjust strategy parameters in real-time
- Visualize performance and risk metrics
- Explore the momentum signal and current positions
- Understand the methodology with built-in explanations

**Deploy online**: Push to GitHub and connect to [Streamlit Cloud](https://streamlit.io/cloud) for free hosting.

## Machine Learning Model

Train the ML regime filter locally:

```bash
# Basic training
python train_model.py

# Force fresh data download
python train_model.py --fresh

# Custom training period
python train_model.py --train-end 2020-12-31
```

The script will:
1. Download latest market data
2. Compute features (momentum, volatility, trend)
3. Train Logistic Regression model
4. Save model + metrics to `models/`
5. Generate training report

Models are saved with timestamps for version tracking.

## Quick Start

```python
from src.data_loader import load_universe_data
from src.signal import compute_momentum_signal
from src.backtest import run_backtest
from src.metrics import compute_metrics

# Load data
prices = load_universe_data()

# Compute momentum signal
signals = compute_momentum_signal(prices)

# Run backtest
results = run_backtest(prices, signals)

# Compute performance metrics
metrics = compute_metrics(results)
print(metrics)
```

## Methodology

### 1. Momentum Signal (12-1)

The classic academic momentum signal, following Jegadeesh & Titman (1993):

```
Momentum(t) = Return(t-12, t-1) = Price(t-1) / Price(t-12) - 1
```

**Why skip the last month?**
Short-term mean reversion contaminates the momentum signal. Excluding the most recent month improves signal quality.

### 2. Portfolio Construction

Each month:
1. Compute momentum score for all stocks
2. Rank stocks cross-sectionally
3. Long top 10, Short bottom 10 (or Long-only: top 10)
4. Equal-weight positions
5. Apply 10 bps transaction cost per trade

### 3. Regime Filter (Enhancement)

**Baseline approach** (SMA200):
- Risk-On: S&P 500 > 200-day SMA → Full exposure
- Risk-Off: S&P 500 < 200-day SMA → Reduced exposure (50% cash)

**Optional ML approach**:
- Logistic Regression classifier
- Features: trend indicators, realized volatility
- No deep learning (interpretability matters)

## Key Metrics

The backtest produces:
- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Turnover
- Transaction cost impact
- Annual returns table

## Research Rigor

This project explicitly addresses:

| Concern | How we handle it |
|---------|------------------|
| Data leakage | Signal computed only with data available at time t |
| Look-ahead bias | Prices at month-end, trades at next month-open |
| Survivorship bias | Acknowledged limitation (DJ30 current constituents) |
| Transaction costs | 10 bps per trade included |
| Overfitting | No parameter optimization, academic standard (12-1) |

## Limitations

1. **Survivorship bias**: Using current DJ30 constituents, not historical
2. **Small universe**: 30 stocks limits diversification
3. **Equal-weight**: May not be optimal, but simple and transparent
4. **Monthly rebalancing**: Daily data could improve precision
5. **No market impact**: Assumes unlimited liquidity (reasonable for DJ30)

## Stress Tests

The backtest includes analysis of:
- **COVID crash (March 2020)**: Rapid drawdown and recovery
- **2022 Bear Market**: Interest rate driven decline

## Future Improvements (Not implemented)

- Expand to S&P 100 or S&P 500
- Add fundamental filters (quality, value)
- Test alternative lookback periods
- Implement risk parity weighting
- Add sector neutrality constraint

## References

- Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers
- Carhart, M. M. (1997). On Persistence in Mutual Fund Performance
- Moskowitz, T. J., & Grinblatt, M. (1999). Do Industries Explain Momentum?
- Faber, M. (2007). A Quantitative Approach to Tactical Asset Allocation

## License

MIT License - See LICENSE file for details

## Author

[Your Name]
ESCP / Polytechnique Candidate

---

*This project is for educational and research purposes only. Past performance does not guarantee future results.*
