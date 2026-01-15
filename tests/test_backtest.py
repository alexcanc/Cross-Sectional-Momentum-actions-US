"""
Unit tests for src/backtest.py

Tests backtesting engine, return computation, and transaction costs.
"""

import pytest
import pandas as pd
import numpy as np

from src.backtest import (
    run_backtest,
    compute_returns,
    compute_turnover,
    compute_drawdowns,
    BacktestResult,
)
from src.config import Config


class TestComputeReturns:
    """Test suite for return computation."""

    def test_returns_shape(self, sample_prices):
        """Test that returns have same shape as prices."""
        returns = compute_returns(sample_prices)

        assert returns.shape == sample_prices.shape

    def test_returns_first_row_nan(self, sample_prices):
        """Test that first row is NaN (no prior price)."""
        returns = compute_returns(sample_prices)

        assert returns.iloc[0].isna().all()

    def test_returns_calculation(self):
        """Test return calculation with known values."""
        dates = pd.date_range(start='2020-01-31', periods=3, freq='ME')
        prices = pd.DataFrame({
            'A': [100, 110, 121],
            'B': [100, 90, 81],
        }, index=dates)

        returns = compute_returns(prices)

        # A: 110/100 - 1 = 0.10, 121/110 - 1 = 0.10
        assert abs(returns.loc[dates[1], 'A'] - 0.10) < 0.001
        assert abs(returns.loc[dates[2], 'A'] - 0.10) < 0.001

        # B: 90/100 - 1 = -0.10, 81/90 - 1 = -0.10
        assert abs(returns.loc[dates[1], 'B'] - (-0.10)) < 0.001
        assert abs(returns.loc[dates[2], 'B'] - (-0.10)) < 0.001


class TestComputeTurnover:
    """Test suite for turnover computation."""

    def test_turnover_no_change(self):
        """Test turnover is zero when weights don't change."""
        dates = pd.date_range(start='2020-01-31', periods=3, freq='ME')

        weights = pd.DataFrame({
            'A': [0.5, 0.5, 0.5],
            'B': [0.5, 0.5, 0.5],
        }, index=dates)

        returns = pd.DataFrame({
            'A': [np.nan, 0.0, 0.0],
            'B': [np.nan, 0.0, 0.0],
        }, index=dates)

        turnover = compute_turnover(weights, returns)

        # After first period, turnover should be ~0
        assert abs(turnover.iloc[1]) < 0.001
        assert abs(turnover.iloc[2]) < 0.001

    def test_turnover_full_change(self):
        """Test turnover when portfolio completely changes."""
        dates = pd.date_range(start='2020-01-31', periods=2, freq='ME')

        weights = pd.DataFrame({
            'A': [1.0, 0.0],
            'B': [0.0, 1.0],
        }, index=dates)

        returns = pd.DataFrame({
            'A': [np.nan, 0.0],
            'B': [np.nan, 0.0],
        }, index=dates)

        turnover = compute_turnover(weights, returns)

        # Full rebalance: |0-1| + |1-0| = 2 (two-way)
        assert abs(turnover.iloc[1] - 2.0) < 0.001


class TestComputeDrawdowns:
    """Test suite for drawdown computation."""

    def test_drawdown_at_peak_is_zero(self):
        """Test that drawdown at peak is zero."""
        dates = pd.date_range(start='2020-01-31', periods=5, freq='ME')
        cumulative = pd.Series([1.0, 1.1, 1.2, 1.15, 1.25], index=dates)

        drawdowns = compute_drawdowns(cumulative)

        # At new highs (index 0, 1, 2, 4), drawdown should be 0
        assert drawdowns.iloc[0] == 0.0
        assert drawdowns.iloc[1] == 0.0
        assert drawdowns.iloc[2] == 0.0
        assert drawdowns.iloc[4] == 0.0

    def test_drawdown_calculation(self):
        """Test drawdown calculation with known values."""
        dates = pd.date_range(start='2020-01-31', periods=4, freq='ME')
        cumulative = pd.Series([1.0, 1.2, 0.96, 1.1], index=dates)

        drawdowns = compute_drawdowns(cumulative)

        # At index 2: (0.96 - 1.2) / 1.2 = -0.2
        assert abs(drawdowns.iloc[2] - (-0.2)) < 0.001

        # At index 3: (1.1 - 1.2) / 1.2 = -0.0833
        assert abs(drawdowns.iloc[3] - (-1/12)) < 0.001

    def test_drawdown_always_nonpositive(self, sample_cumulative_returns):
        """Test that drawdowns are always <= 0."""
        drawdowns = compute_drawdowns(sample_cumulative_returns)

        assert (drawdowns <= 0).all()


class TestRunBacktest:
    """Test suite for full backtest execution."""

    def test_backtest_returns_correct_type(self, sample_prices, sample_benchmark):
        """Test that backtest returns BacktestResult."""
        config = Config()
        config.N_LONG = 2
        config.N_SHORT = 2

        result = run_backtest(sample_prices, sample_benchmark, config)

        assert isinstance(result, BacktestResult)

    def test_backtest_result_components(self, sample_prices, sample_benchmark):
        """Test that backtest result has all required components."""
        config = Config()
        config.N_LONG = 2
        config.N_SHORT = 2

        result = run_backtest(sample_prices, sample_benchmark, config)

        # Check all attributes exist
        assert hasattr(result, 'strategy_returns')
        assert hasattr(result, 'benchmark_returns')
        assert hasattr(result, 'cumulative_returns')
        assert hasattr(result, 'cumulative_benchmark')
        assert hasattr(result, 'weights')
        assert hasattr(result, 'turnover')
        assert hasattr(result, 'transaction_costs')

    def test_backtest_cumulative_starts_at_one(self, sample_prices, sample_benchmark):
        """Test that cumulative returns start near 1."""
        config = Config()
        config.N_LONG = 2
        config.N_SHORT = 2

        result = run_backtest(sample_prices, sample_benchmark, config)

        # First cumulative value should be close to 1 + first return
        assert abs(result.cumulative_returns.iloc[0] - (1 + result.strategy_returns.iloc[0])) < 0.01

    def test_backtest_transaction_costs_applied(self, sample_prices, sample_benchmark):
        """Test that transaction costs are applied."""
        config = Config()
        config.N_LONG = 2
        config.N_SHORT = 2
        config.TRANSACTION_COST_BPS = 10.0

        result = run_backtest(sample_prices, sample_benchmark, config)

        # Transaction costs should be non-zero (we have turnover)
        assert result.transaction_costs.sum() > 0

        # Net returns should be less than gross returns (due to costs)
        gross_cum = (1 + result.strategy_returns_gross).prod()
        net_cum = (1 + result.strategy_returns).prod()

        assert net_cum <= gross_cum

    def test_backtest_long_only(self, sample_prices, sample_benchmark):
        """Test long-only backtest variant."""
        config = Config()
        config.N_LONG = 2
        config.N_SHORT = 0
        config.PORTFOLIO_TYPE = "long_only"

        result = run_backtest(sample_prices, sample_benchmark, config)

        # Weights should all be non-negative
        assert (result.weights >= 0).all().all()

    def test_backtest_with_regime_filter(self, sample_prices, sample_benchmark):
        """Test backtest with regime filter applied."""
        config = Config()
        config.N_LONG = 2
        config.N_SHORT = 2
        config.RISK_OFF_EXPOSURE = 0.5

        # Create simple regime signal (alternating)
        regime = pd.Series(
            [1, 0] * (len(sample_prices) // 2),
            index=sample_prices.index[:len(sample_prices)]
        )

        result = run_backtest(sample_prices, sample_benchmark, config, regime_signal=regime)

        # Result should still be valid
        assert isinstance(result, BacktestResult)
        assert len(result.strategy_returns) > 0
