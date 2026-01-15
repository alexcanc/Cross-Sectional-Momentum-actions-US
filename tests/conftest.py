"""
Pytest fixtures for Cross-Sectional Momentum Strategy tests.

These fixtures provide consistent test data across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_prices():
    """
    Generate sample price data for testing.

    Creates a DataFrame with 5 stocks over 24 months with realistic price movements.
    """
    np.random.seed(42)

    dates = pd.date_range(start='2020-01-31', periods=24, freq='ME')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    # Generate random returns and cumulate to prices
    returns = np.random.randn(24, 5) * 0.05  # 5% monthly vol

    # Add some momentum: first 2 stocks trend up, last 2 trend down
    returns[:, 0] += 0.02  # AAPL trending up
    returns[:, 1] += 0.015  # MSFT trending up
    returns[:, 3] -= 0.01  # AMZN trending down
    returns[:, 4] -= 0.015  # META trending down

    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    df = pd.DataFrame(prices, index=dates, columns=tickers)

    return df


@pytest.fixture
def sample_benchmark():
    """
    Generate sample benchmark data for testing.

    Creates SPY price series aligned with sample_prices.
    """
    np.random.seed(123)

    dates = pd.date_range(start='2020-01-31', periods=24, freq='ME')

    returns = np.random.randn(24) * 0.04 + 0.005  # Positive drift
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({'SPY': prices}, index=dates)

    return df


@pytest.fixture
def sample_benchmark_daily():
    """
    Generate daily benchmark data for regime filter testing.

    Creates ~500 days of SPY data with realistic patterns.
    """
    np.random.seed(456)

    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')

    returns = np.random.randn(500) * 0.01 + 0.0003  # Small positive drift

    # Add regime changes: bearish period in middle
    returns[150:250] -= 0.002  # Bearish period

    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({'SPY': prices}, index=dates)

    return df


@pytest.fixture
def sample_returns():
    """
    Generate sample return series for metrics testing.
    """
    np.random.seed(789)

    dates = pd.date_range(start='2020-01-31', periods=36, freq='ME')

    # Generate returns with known characteristics
    returns = np.random.randn(36) * 0.03 + 0.008  # ~8% annual return, 10% vol

    # Add some negative months for drawdown
    returns[10:13] = [-0.05, -0.08, -0.03]  # Drawdown period

    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_cumulative_returns(sample_returns):
    """
    Compute cumulative returns from sample returns.
    """
    return (1 + sample_returns).cumprod()


@pytest.fixture
def sample_momentum():
    """
    Generate sample momentum scores for portfolio construction tests.
    """
    np.random.seed(111)

    dates = pd.date_range(start='2020-01-31', periods=12, freq='ME')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
               'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']

    # Create momentum with clear ranking
    base_momentum = np.array([0.3, 0.25, 0.2, 0.15, 0.1,
                              -0.05, -0.1, -0.15, -0.2, -0.25])

    # Add some noise across time
    momentum = np.tile(base_momentum, (12, 1))
    momentum += np.random.randn(12, 10) * 0.02

    df = pd.DataFrame(momentum, index=dates, columns=tickers)

    return df
